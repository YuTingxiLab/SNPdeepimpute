"""Data loading and genotype conversion utilities."""

from typing import Union
import gzip
import logging
import os
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


SUPPORTED_FILE_FORMATS = {"vcf", "csv", "tsv"}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pprint(text: str) -> None:
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")


class DataReader:
    """Handle reference/target VCF or tabular genotype files."""

    def __init__(self) -> None:
        self.target_is_gonna_be_phased = None
        self.target_set = None
        self.target_sample_value_index = 2
        self.ref_sample_value_index = 2
        self.target_file_extension = None
        self.allele_count = 2
        self.genotype_vals = None
        self.ref_is_phased = None
        self.reference_panel = None
        self.VARIANT_COUNT = 0
        self.is_phased = False
        self.MISSING_VALUE = None
        self.ref_is_hap = False
        self.target_is_hap = False
        self.ref_n_header_lines = []
        self.ref_n_data_header = ""
        self.target_n_header_lines = []
        self.target_n_data_header = ""
        self.ref_separator = None
        self.delimiter_dictionary = {"vcf": "\t", "csv": ",", "tsv": "\t", "infer": "\t"}
        self.ref_file_extension = "vcf"
        self.test_file_extension = "vcf"
        self.target_is_phased = True

    def __read_csv(self, file_path: str, is_vcf: bool = False, is_reference: bool = False,
                   separator: str = "\t", first_column_is_index: bool = True,
                   comments: str = "##") -> pd.DataFrame:
        pprint("Reading the file...")
        data_header = None
        line_counter = 0
        root, ext = os.path.splitext(file_path)
        opener = gzip.open if ext == '.gz' else open
        with opener(file_path, 'rt') as f_in:
            while True:
                line = f_in.readline()
                if line.startswith(comments):
                    line_counter += 1
                    if is_reference:
                        self.ref_n_header_lines.append(line)
                    else:
                        self.target_n_header_lines.append(line)
                else:
                    data_header = line
                    break
        if data_header is None:
            raise IOError("The file only contains comments!")
        df = pd.read_csv(file_path, sep=separator, comment=comments, skiprows=line_counter)
        if first_column_is_index:
            df.set_index(df.columns[0], inplace=True)
        return df

    def __find_file_extension(self, file_path: str, file_format: str, delimiter: str):
        separator = "\t"
        found_file_format = None
        if file_format not in ["infer"] + list(SUPPORTED_FILE_FORMATS):
            raise ValueError("File extension must be one of {'vcf', 'csv', 'tsv', 'infer'}.")
        if file_format == 'infer':
            tokens = file_path.split('.')
            for possible_extension in tokens[::-1]:
                if possible_extension in SUPPORTED_FILE_FORMATS:
                    found_file_format = possible_extension
                    separator = self.delimiter_dictionary[possible_extension] if delimiter is None else delimiter
                    break
            if found_file_format is None:
                logging.warning("Could not infer the file type. Using tsv as the last resort.")
                found_file_format = "tsv"
        else:
            found_file_format = file_format
            separator = self.delimiter_dictionary[file_format] if delimiter is None else delimiter
        return found_file_format, separator

    def assign_training_set(self, file_path: str,
                            target_is_gonna_be_phased_or_haps: bool,
                            variants_as_columns: bool = False,
                            delimiter=None,
                            file_format: str = "infer",
                            first_column_is_index: bool = True,
                            comments: str = "##") -> None:
        self.target_is_gonna_be_phased = target_is_gonna_be_phased_or_haps
        self.ref_file_extension, self.ref_separator = self.__find_file_extension(file_path, file_format, delimiter)
        if file_format == "infer":
            pprint(f"Ref file format is {self.ref_file_extension}.")
        self.reference_panel = self.__read_csv(file_path, is_reference=True, is_vcf=False, separator=self.ref_separator,
                                               first_column_is_index=first_column_is_index,
                                               comments=comments) if self.ref_file_extension != 'vcf' else self.__read_csv(
            file_path, is_reference=True, is_vcf=True, separator='\t', first_column_is_index=False, comments="##")
        if self.ref_file_extension != "vcf":
            if variants_as_columns:
                self.reference_panel = self.reference_panel.transpose()
            self.reference_panel.reset_index(drop=False, inplace=True)
            self.reference_panel.rename(columns={self.reference_panel.columns[0]: "ID"}, inplace=True)
        else:
            self.ref_sample_value_index += 8
        self.ref_is_hap = not ("|" in self.reference_panel.iloc[0, self.ref_sample_value_index] or "/" in
                               self.reference_panel.iloc[0, self.ref_sample_value_index])
        self.ref_is_phased = "|" in self.reference_panel.iloc[0, self.ref_sample_value_index]
        if self.ref_is_hap and not target_is_gonna_be_phased_or_haps:
            raise ValueError("The reference contains haploids while the target will be unphased diploids.")
        if not (self.ref_is_phased or self.ref_is_hap) and target_is_gonna_be_phased_or_haps:
            raise ValueError("The reference contains unphased diploids while the target will be phased or haploid data.")
        self.VARIANT_COUNT = self.reference_panel.shape[0]
        pprint(f"{self.reference_panel.shape[1] - (self.ref_sample_value_index - 1)} "
               f"{'haploid' if self.ref_is_hap else 'diploid'} samples with {self.VARIANT_COUNT} variants found!")
        self.is_phased = target_is_gonna_be_phased_or_haps and (self.ref_is_phased or self.ref_is_hap)
        original_sep = "|" if self.ref_is_phased or self.ref_is_hap else "/"
        final_sep = "|" if self.is_phased else "/"

        def get_diploid_allels(genotype_vals):
            allele_set = set()
            for genotype_val in genotype_vals:
                v1, v2 = genotype_val.split(final_sep)
                allele_set.update([v1, v2])
            return np.array(list(allele_set))

        genotype_vals = pd.unique(self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].values.ravel('K'))
        if self.ref_is_phased and not target_is_gonna_be_phased_or_haps:
            phased_to_unphased = {}
            for i in range(genotype_vals.shape[0]):
                key = genotype_vals[i]
                v1, v2 = [int(s) for s in genotype_vals[i].split(original_sep)]
                genotype_vals[i] = f"{min(v1, v2)}/{max(v1, v2)}"
                phased_to_unphased[key] = genotype_vals[i]
            self.reference_panel.iloc[:, self.ref_sample_value_index - 1:].replace(phased_to_unphased, inplace=True)

        self.genotype_vals = np.unique(genotype_vals)
        self.alleles = get_diploid_allels(self.genotype_vals) if not self.ref_is_hap else self.genotype_vals
        self.allele_count = len(self.alleles)
        self.MISSING_VALUE = self.allele_count if self.is_phased else len(self.genotype_vals)
        if self.is_phased:
            self.hap_map = {str(v): i for i, v in enumerate(list(sorted(self.alleles)))}
            self.hap_map.update({'.': self.MISSING_VALUE})
            self.r_hap_map = {i: k for k, i in self.hap_map.items()}
            self.map_preds_2_allele = np.vectorize(lambda x: self.r_hap_map[x])
        else:
            unphased_missing_genotype = "./."
            self.replacement_dict = {g: i for i, g in enumerate(list(sorted(self.genotype_vals)))}
            self.replacement_dict[unphased_missing_genotype] = self.MISSING_VALUE
            self.reverse_replacement_dict = {v: k for k, v in self.replacement_dict.items()}
        self.SEQ_DEPTH = self.allele_count + 1 if self.is_phased else len(self.genotype_vals)
        pprint("Done!")

    def assign_test_set(self, file_path: str,
                        variants_as_columns: bool = False,
                        delimiter=None,
                        file_format: str = "infer",
                        first_column_is_index: bool = True,
                        comments: str = "##") -> None:
        if self.reference_panel is None:
            raise RuntimeError("First assign a training set using 'assign_training_set'.")
        self.target_file_extension, separator = self.__find_file_extension(file_path, file_format, delimiter)
        test_df = self.__read_csv(file_path, is_reference=False, is_vcf=False, separator=separator,
                                  first_column_is_index=first_column_is_index,
                                  comments=comments) if self.ref_file_extension != 'vcf' else self.__read_csv(file_path,
                                                                                                              is_reference=False,
                                                                                                              is_vcf=True,
                                                                                                              separator='\t',
                                                                                                              first_column_is_index=False,
                                                                                                              comments="##")
        if self.target_file_extension != "vcf":
            if variants_as_columns:
                test_df = test_df.transpose()
            test_df.reset_index(drop=False, inplace=True)
            test_df.rename(columns={test_df.columns[0]: "ID"}, inplace=True)
        else:
            self.target_sample_value_index += 8
        self.target_is_hap = not ("|" in test_df.iloc[0, self.target_sample_value_index] or "/" in test_df.iloc[0, self.target_sample_value_index])
        is_phased = "|" in test_df.iloc[0, self.target_sample_value_index]
        test_var_count = test_df.shape[0]
        pprint(f"{test_var_count} {'haplotype' if self.target_is_hap else 'diplotype'} variants found!")
        if (self.target_is_hap or is_phased) and not (self.ref_is_phased or self.ref_is_hap):
            raise RuntimeError("The training set contains unphased data. The target must be unphased as well.")
        if self.ref_is_hap and not (self.target_is_hap or is_phased):
            raise RuntimeError("The training set contains haploids. The target must be phased or haploids as well.")
        self.target_set = test_df.merge(right=self.reference_panel["ID"], on='ID', how='right')
        if self.target_file_extension == "vcf" == self.ref_file_extension:
            self.target_set[self.reference_panel.columns[:9]] = self.reference_panel[self.reference_panel.columns[:9]]
        self.target_set = self.target_set.astype('str')
        self.target_set.fillna('.' if self.target_is_hap else '.|.' if self.is_phased else './.', inplace=True)
        self.target_set.replace('nan', '.' if self.target_is_hap else '.|.' if self.is_phased else './.', inplace=True)
        pprint("Done!")

    def __map_hap_2_ind_parent_1(self, x) -> int:
        return self.hap_map[x.split('|')[0]]

    def __map_hap_2_ind_parent_2(self, x) -> int:
        return self.hap_map[x.split('|')[1]]

    def __map_haps_2_ind(self, x) -> int:
        return self.hap_map[x]

    def __diploids_to_hap_vecs(self, data: pd.DataFrame) -> np.ndarray:
        _x = np.empty((data.shape[1] * 2, data.shape[0]), dtype=np.int32)
        _x[0::2] = np.vectorize(self.__map_hap_2_ind_parent_1)(data.values.T)
        _x[1::2] = np.vectorize(self.__map_hap_2_ind_parent_2)(data.values.T)
        return _x

    def __get_forward_data(self, data: pd.DataFrame) -> np.ndarray:
        if self.is_phased:
            is_haps = "|" not in data.iloc[0, 0]
            if not is_haps:
                return self.__diploids_to_hap_vecs(data)
            else:
                return np.vectorize(self.__map_haps_2_ind)(data.values.T)
        else:
            return data.replace(self.replacement_dict).values.T.astype(np.int32)

    def get_ref_set(self, starting_var_index=0, ending_var_index=0) -> np.ndarray:
        if 0 <= starting_var_index < ending_var_index:
            return self.__get_forward_data(
                data=self.reference_panel.iloc[starting_var_index:ending_var_index, self.ref_sample_value_index - 1:])
        else:
            pprint("No variant indices provided or indices not valid, using the whole sequence...")
            return self.__get_forward_data(data=self.reference_panel.iloc[:, self.ref_sample_value_index - 1:])

    def get_target_set(self, starting_var_index=0, ending_var_index=0) -> np.ndarray:
        if 0 <= starting_var_index < ending_var_index:
            return self.__get_forward_data(
                data=self.target_set.iloc[starting_var_index:ending_var_index, self.target_sample_value_index - 1:])
        else:
            pprint("No variant indices provided or indices not valid, using the whole sequence...")
            return self.__get_forward_data(data=self.target_set.iloc[:, self.target_sample_value_index - 1:])

    def __convert_hap_probs_to_diploid_genotypes(self, allele_probs: np.ndarray) -> np.ndarray:
        n_haploids, n_variants, n_alleles = allele_probs.shape
        if n_haploids % 2 != 0:
            raise ValueError("Number of haploids should be even.")
        if n_alleles == 2:
            print("Outputting GP in predictions.")
        n_samples = n_haploids // 2
        genotypes = np.empty((n_samples, n_variants), dtype=object)
        haploids_as_diploids = allele_probs.reshape((n_samples, 2, n_variants, -1))
        variant_genotypes = np.vectorize(self.map_preds_2_allele)(np.argmax(haploids_as_diploids, axis=-1))

        def process_variant_in_sample(haps_for_sample_at_variant, variant_genotypes_for_sample_at_variant):
            if n_alleles > 2:
                return '|'.join(variant_genotypes_for_sample_at_variant)
            else:
                phased_probs = np.outer(haps_for_sample_at_variant[0], haps_for_sample_at_variant[1]).flatten()
                unphased_probs = np.array([phased_probs[0], phased_probs[1] + phased_probs[2], phased_probs[-1]])
                unphased_probs_str = ','.join([f"{v:.6f}" for v in unphased_probs])
                alt_dosage = np.dot(unphased_probs, [0, 1, 2])
                return '|'.join(variant_genotypes_for_sample_at_variant) + f":{unphased_probs_str}:{alt_dosage:.3f}"

        def process_sample(i):
            return np.array([
                process_variant_in_sample(haploids_as_diploids[i, :, j, :], variant_genotypes[i, :, j])
                for j in range(n_variants)
            ])

        genotypes = Parallel(n_jobs=-1)(delayed(process_sample)(i) for i in tqdm(range(n_samples)))
        return np.array(genotypes)

    def __convert_hap_probs_to_hap_genotypes(self, allele_probs: np.ndarray) -> np.ndarray:
        return np.argmax(allele_probs, axis=1).astype(str)

    def __convert_unphased_probs_to_genotypes(self, allele_probs: np.ndarray) -> np.ndarray:
        n_samples, n_variants, n_alleles = allele_probs.shape
        genotypes = np.zeros((n_samples, n_variants), dtype=object)
        for i in tqdm(range(n_samples)):
            for j in range(n_variants):
                unphased_probs = allele_probs[i, j]
                variant_genotypes = np.vectorize(self.reverse_replacement_dict.get)(
                    np.argmax(unphased_probs, axis=-1)).flatten()
                genotypes[i, j] = variant_genotypes
        return genotypes

    def preds_to_genotypes(self, predictions: Union[str, np.ndarray]) -> pd.DataFrame:
        if isinstance(predictions, str):
            preds = np.load(predictions)
        else:
            preds = predictions
        target_df = self.target_set.copy()
        if not self.is_phased:
            target_df[target_df.columns[self.target_sample_value_index - 1:]] = self.__convert_unphased_probs_to_genotypes(preds).T
        elif self.target_is_hap:
            target_df[target_df.columns[self.target_sample_value_index - 1:]] = self.__convert_hap_probs_to_hap_genotypes(preds).T
        else:
            pred_format = "GT:GP:DS" if preds.shape[-1] == 2 else "GT"
            target_df = self.__convert_genotypes_to_vcf(self.__convert_hap_probs_to_diploid_genotypes(preds).T, pred_format)
        return target_df

    def __get_headers_for_output(self, contain_probs: bool, chr: int = 22):
        headers = ["##fileformat=VCFv4.2",
                   "##source=DeepImpute-pytorch",
                   "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated Alternate Allele Frequency\">",
                   "##INFO=<ID=MAF,Number=1,Type=Float,Description=\"Estimated Minor Allele Frequency\">",
                   "##INFO=<ID=AVG_CS,Number=1,Type=Float,Description=\"Average Call Score\">",
                   "##INFO=<ID=IMPUTED,Number=0,Type=Flag,Description=\"Marker was imputed\">",
                   "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                   ]
        probs_headers = [
            "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Estimated Alternate Allele Dosage : [P(0/1)+2*P(1/1)]\">",
            "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Estimated Posterior Probabilities for Genotypes 0/0, 0/1 and 1/1\">"]
        if contain_probs:
            headers.extend(probs_headers)
        return headers

    def __convert_genotypes_to_vcf(self, genotypes, pred_format="GT:GP:DS"):
        new_vcf = self.target_set.copy()
        new_vcf[new_vcf.columns[self.target_sample_value_index - 1:]] = genotypes
        new_vcf["FORMAT"] = pred_format
        new_vcf["QUAL"] = "."
        new_vcf["FILTER"] = "."
        new_vcf["INFO"] = "IMPUTED"
        return new_vcf

    def write_ligated_results_to_file(self, df: pd.DataFrame, file_name: str, compress: bool = True) -> str:
        to_write_format = self.ref_file_extension
        with gzip.open(f"{file_name}.{to_write_format}.gz", 'wt') if compress else open(
                f"{file_name}.{to_write_format}", 'wt') as f_out:
            if self.ref_file_extension == "vcf":
                f_out.write("\n".join(self.__get_headers_for_output(contain_probs="GP" in df["FORMAT"].values[0])) + "\n")
            else:
                f_out.write("\n".join(self.ref_n_header_lines))
        df.to_csv(f"{file_name}.{to_write_format}.gz" if compress else f"{file_name}.{to_write_format}",
                  sep=self.ref_separator, mode='a', index=False)
        return f"{file_name}.{to_write_format}.gz" if compress else f"{file_name}.{to_write_format}"


def calculate_maf(genotype_array: np.ndarray) -> np.ndarray:
    allele_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=0, arr=genotype_array)
    total_alleles = 2 * genotype_array.shape[0]
    minor_allele_counts = 2 * allele_counts[2] + allele_counts[1]
    maf = minor_allele_counts / total_alleles
    return maf


def remove_similar_rows(array: np.ndarray) -> np.ndarray:
    print("Finding duplicate haploids in training set.")
    unique_array = np.unique(array, axis=0)
    print(f"Removed {len(array) - len(unique_array)} rows. {len(unique_array)} training samples remaining.")
    return unique_array


def create_directories(save_dir: str, models_dir: str = "models", outputs: str = "out") -> None:
    for dd in [save_dir, f"{save_dir}/{models_dir}", f"{save_dir}/{outputs}"]:
        if not os.path.exists(dd):
            os.makedirs(dd)


def clear_dir(path: str) -> None:
    if os.path.exists(path):
        for entry in os.scandir(path):
            if entry.is_dir():
                clear_dir(entry.path)
            else:
                os.remove(entry.path)
        os.rmdir(path)


def load_chunk_info(save_dir: str, break_points):
    chunk_info = {ww: False for ww in list(range(len(break_points) - 1))}
    info_path = os.path.join(save_dir, "models", "chunks_info.json")
    if os.path.isfile(info_path):
        with open(info_path, 'r') as f:
            loaded_chunks_info = json.load(f)
            if isinstance(loaded_chunks_info, dict) and len(loaded_chunks_info) == len(chunk_info):
                pprint("Resuming the training...")
                chunk_info = {int(k): v for k, v in loaded_chunks_info.items()}
    return chunk_info


def save_chunk_status(save_dir: str, chunk_info) -> None:
    info_path = os.path.join(save_dir, "models", "chunks_info.json")
    with open(info_path, "w") as outfile:
        json.dump(chunk_info, outfile)
