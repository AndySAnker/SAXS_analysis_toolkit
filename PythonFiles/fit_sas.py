import os
import argparse
import time
from sklearn.metrics import accuracy_score
from SAXS_analysis.utils.configs import load_config
from SAXS_analysis.fitting.fit_sas import SAS_Fitter
from SAXS_analysis.data_processing.utils import load_hdf5_data
from SAXS_analysis.utils.logging import setup_logging
from SAXS_analysis.utils.constants import ROOT_DIR
class Result:
    def __init__(self, formfactor, goodness_of_fit, R_w, index, logger):
        self.formfactor = formfactor
        self.goodness_of_fit = goodness_of_fit
        self.R_w = R_w
        self.index = index
        logger.info(f"Result: formfactor: {self.formfactor}, goodness_of_fit: {self.goodness_of_fit}, R_w: {self.R_w}, index: {self.index}")

def fit_sas_data_for_formfactors(sas_fitter, formfactors, solver, smearing, logger):
    results = []
    logger.debug(f"Fitting for formfactors: {formfactors}")
    for formfactor in formfactors:
        formfactor = str(formfactor)
        logger.debug(f"Fitting for formfactor: {formfactor}")
        try:
            goodness_of_fit, R_w, fitted_params = sas_fitter.fit_sas_data(formfactor, solver, smearing)
            logger.debug(f"Fitted formfactor: {formfactor}, goodness_of_fit: {goodness_of_fit}, R_w: {R_w}")
            results.append(Result(formfactor, goodness_of_fit, R_w, sas_fitter.current_index, logger))
        except Exception as e:
            logger.error(f"Error fitting {formfactor}: {e}")
    return results

def analyze_sas_data(sas_fitter, datafiles, qmin, qmax, error_weighting, normalization_type, form_factors, solver, smearing, logger):
    all_results = []
    for index, datafile in enumerate(datafiles):
        import pdb; pdb.set_trace()
        logger.info(f"Fitting datafile {index} of {len(datafiles)}")
        sas_fitter.load_data(datafile, qmin, qmax, error_weighting, normalization_type)
        logger.debug(f"Loaded datafile {index} of {len(datafiles)}")
        sas_fitter.current_index = index
        logger.debug(f"Current index: {sas_fitter.current_index}")
        results = fit_sas_data_for_formfactors(sas_fitter, form_factors, solver, smearing, logger)
        logger.debug(f"Fitted datafile {index} of {len(datafiles)}")
        all_results.extend(results)
    return all_results

def get_best_fits(results, criterion):
    best_fits = []
    current_best = None
    current_index = None
    
    for result in results:
        if current_index != result.index:
            if current_best is not None:
                best_fits.append(current_best.formfactor)
            current_index = result.index
            current_best = result
        else:
            if criterion == 'goodness_of_fit' and result.goodness_of_fit < current_best.goodness_of_fit:
                current_best = result
            elif criterion == 'R_w' and result.R_w < current_best.R_w:
                current_best = result
    
    if current_best is not None:
        best_fits.append(current_best.formfactor)
    
    return best_fits

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        logger = setup_logging(config)

        logger.info(f"Starting experiment {config['experiment_id']}: {config['experiment_name']}")
        logger.info(f'Configuration: {config}')

        logger.debug(f"data_dir: {config['data_dir']}")
        logger.debug(f"DataName: {config['DataName']}")
        logger.debug(f"NumFiles: {config['NumFiles']}")
        logger.debug(f"qmin: {config['qmin']}")
        logger.debug(f"qmax: {config['qmax']}")
        logger.debug(f"error_weighting: {config['error_weighting']}")
        logger.debug(f"normalise_data: {config['normalise_data']}")
        logger.debug(f"form_factors: {config['form_factors']}")
        logger.debug(f"solver: {config['solver']}")
        logger.debug(f"smearing: {config['smearing']}")

        # Load and process data
        logger.info("Loading and processing data...")
        datafiles, y_decoded = load_hdf5_data(os.path.join(ROOT_DIR, config['data_dir'], config['DataName']), config['NumFiles'], config['qmin'], config['qmax'])
        logger.info(f'Datafiles: {datafiles}')

        # Fit SAS data
        logger.info("Fitting SAS data...")
        sas_fitter = SAS_Fitter(os.path.join(ROOT_DIR, config['data_dir'], config['DataName']))
        logger.info(f'sas_fitter: {sas_fitter}')
        fitting_results = analyze_sas_data(sas_fitter, datafiles, config['qmin'], config['qmax'], config['error_weighting'], config['normalise_data'], config['form_factors'], config['solver'], config['smearing'], logger)
        
        # Analyze results
        logger.info("Analyzing results...")
        
        # Get best fits
        predicted_formfactors_goodness_of_fit = get_best_fits(fitting_results, 'goodness_of_fit')
        predicted_formfactors_R_w = get_best_fits(fitting_results, 'R_w')
        
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