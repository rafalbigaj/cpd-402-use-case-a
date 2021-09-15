# Uses Python library "logging", writing logs into $BATCH_OUTPUT_DIR/evaluation.log
# WMl will store the output dir as a .zip file in a new data asset.

import os
import shutil
import sys
import traceback
import logging

input_dir = os.environ.get("BATCH_INPUT_DIR")
output_dir = os.environ.get("BATCH_OUTPUT_DIR")

# The Job log in WML does not include stdout or stderr
# Therefore, write tracing info into a file in BATCH_OUTPUT_DIR
#
if output_dir:
    logfile = os.path.join(output_dir, "evaluation.log")
    logging.basicConfig(filename=logfile, level=logging.INFO)
else:
    # Notebook configuration
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # logging.DEBUG
    input_dir = '.'
    output_dir = '/tmp'

try:
    logging.info("Start of script execution {} {}".format(input_dir, output_dir))

    logging.info("__name__=" + __name__)

    import json
    import pandas as pdf
    from sklearn import metrics
    from ibm_watson_machine_learning import APIClient
    from ibm_watson_machine_learning.deployment import Batch

    logging.info("\n==== Environment ============================")

    for item, value in os.environ.items():
        logging.info('{}: {}'.format(item, value))

    logging.info("getcwd = " + str(os.getcwd()))

    input_file_path = os.getenv("JOBS_PAYLOAD_FILE")

    if input_file_path:
        shutil.copy(input_file_path, output_dir)

    space_id = os.environ["SPACE_ID"]
    deployment_id = os.environ["DEPLOYMENT_ID"]

    wml_credentials = {
        "url": os.getenv(
            "RUNTIME_ENV_APSX_URL", "https://internal-nginx-svc:12443"
        ),
        "token": os.environ["USER_ACCESS_TOKEN"],
        "instance_id": "openshift",
        "version": "4.0",
    }

    wml_client = APIClient(wml_credentials)
    wml_client.set.default_space(space_id)

    from ibm_watson_machine_learning.deployment import Batch

    service_batch = Batch(
        source_wml_credentials=wml_credentials,
        source_space_id=space_id,
        target_wml_credentials=wml_credentials,
        target_space_id=space_id
    )

    service_batch.get(deployment_id)

    input_df = pdf.read_csv(input_file_path)

    y_true = input_df.pop('Risk')

    scoring_params = service_batch.run_job(
        payload=input_df,
        background_mode=False)

    predictions = scoring_params['entity']['scoring'].get('predictions')
    if predictions:
        y_pred = pdf.DataFrame(data=predictions[0]['values'], columns=predictions[0]['fields'])['prediction']
        y_proba = pdf.DataFrame(data=predictions[0]['values'], columns=predictions[0]['fields'])['probability']

    ny_true = [y == 'No Risk' for y in y_true]
    ny_pred = [y == 'No Risk' for y in y_pred]

    results = {}

    results['accuracy_score'] = metrics.accuracy_score(ny_true, ny_pred)
    results['precision_score'] = metrics.precision_score(ny_true, ny_pred)
    results['f1_score'] = metrics.f1_score(ny_true, ny_pred)
    results['roc_auc_score'] = metrics.roc_auc_score(ny_true, ny_pred)
    results['log_loss'] = metrics.log_loss(ny_true, [y[0] for y in y_proba])

    logging.info("\n==== Results ============================")

    for metric_name, value in results.items():
        logging.info('{}: {}'.format(metric_name, value))

    expectations = {
        'accuracy_score': 0.8,
        'precision_score': 0.8,
        'f1_score': 0.85,
        'roc_auc_score': 0.7,
        'log_loss': 0.5
    }

    logging.info("\n==== Expectations ============================")

    for metric_name, value in expectations.items():
        logging.info('{}: {}'.format(metric_name, value))

    errors = []

    for metric_name, exp_value in expectations.items():
        value = results[metric_name]
        if metric_name == 'log_loss':
            if value > exp_value:
                errors.append('{} value: {} expected to be lower than {}'.format(metric_name, value, exp_value))
        else:
            if value < exp_value:
                errors.append('{} value: {} expected to be gather than {}'.format(metric_name, value, exp_value))

    logging.info("\n==== Errors ============================")

    for error in errors:
        logging.error(error)

    logging.info("\n==== Files ============================")

    logging.info("writing files to output dir")

    with open(os.path.join(output_dir, "results.json"), "w") as fo:
        json.dump(results, fo, indent=2)

    with open(os.path.join(output_dir, "expectations.json"), "w") as fo:
        json.dump(expectations, fo, indent=2)

    with open(os.path.join(output_dir, "errors.json"), "w") as fo:
        json.dump(errors, fo, indent=2)

    with open(os.path.join(output_dir, "results.txt"), "w") as fo:
        if len(errors) == 0:
            print("Model validation succedded!", file=fo)

            for metric_name, value in results.items():
                print('{}: {}'.format(metric_name, value), file=fo)

        else:
            print("Model validation failed!", file=fo)

            for error in errors:
                print(error, file=fo)

    logging.info("End of script execution")

except Exception as ex:
    logging.error(traceback.format_exc())
    exit(1)

finally:
    logging.info("shutdown")
    logging.shutdown()
