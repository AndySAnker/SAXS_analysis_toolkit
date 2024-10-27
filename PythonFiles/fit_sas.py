import os
import argparse
import time
from tqdm import tqdm
from src.config import Config
from src.fitting.sas_fitter import SASFitter
from src.analysis.result_analyzer import ResultAnalyzer
from sklearn.metrics import accuracy_score
from src.data_processing.utils import load_and_process_data
from src.utils.logging import setup_logging

class Result:
    def __init__(self, formfactor, goodness_of_fit, R_w, index):
        self.formfactor = formfactor
        self.goodness_of_fit = goodness_of_fit
        self.R_w = R_w
        self.index = index

def fit_sas_data_for_formfactors(sas_fitter, formfactors, solver, smearing):
    results = []
    for formfactor in formfactors:
        try:
            goodness_of_fit, R_w, fitted_params = sas_fitter.fit_sas_data(formfactor, solver, smearing)
            results.append(Result(formfactor, goodness_of_fit, R_w, sas_fitter.current_index))
        except Exception as e:
            logger.error(f"Error fitting {formfactor}: {e}")
    return results

def analyze_sas_data(sas_fitter, datafiles, config):
    all_results = []
    for index, datafile in enumerate(tqdm(datafiles, desc="Analyzing datafiles")):
        sas_fitter.load_data(datafile, config.qmin, config.qmax, config.error_weighting, config.normalization_type)
        sas_fitter.current_index = index
        results = fit_sas_data_for_formfactors(sas_fitter, config.form_factors, config.solver, config.smearing)
        all_results.extend(results)
    return all_results

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    try:
        config = Config.from_file(args.config)
        logger = setup_logging(config)

        logger.info(f"Starting experiment {config.experiment_id}: {config.experiment_name}")
        logger.info(f'Configuration: {config}')

        datafiles, y_decoded = load_and_process_data(config.DataName, config.NumFiles, config.qmin, config.qmax)
        logger.info(f'Datafiles: {datafiles}')

        sas_fitter = SASFitter(os.path.join(config.data_dir, config.DataName))
        
        fitting_results = analyze_sas_data(sas_fitter, datafiles, config)
        
        analyzer = ResultAnalyzer(fitting_results, config.NumFiles)
        analyzed_results = analyzer.analyze()
        logger.info(f'Analyzed results: {analyzed_results}')
        
        predicted_formfactors_goodness_of_fit = analyzer.get_best_fits('goodness_of_fit')
        predicted_formfactors_R_w = analyzer.get_best_fits('R_w')
        
        logger.info(f'y_decoded: {y_decoded}')
        logger.info(f'predicted_formfactors_goodness_of_fit: {predicted_formfactors_goodness_of_fit}')
        logger.info(f'predicted_formfactors_R_w: {predicted_formfactors_R_w}')
        
        accuracy_goodness_of_fit = accuracy_score(y_decoded, predicted_formfactors_goodness_of_fit)
        accuracy_R_w = accuracy_score(y_decoded, predicted_formfactors_R_w)
        logger.info(f"Accuracy (goodness-of-fit): {accuracy_goodness_of_fit}")
        logger.info(f"Accuracy (R_w): {accuracy_R_w}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        logger.info(f'Time taken: {(time.time() - start_time)/60:.2f} minutes')

if __name__ == '__main__':
    main()