import os
import pandas as pd
import fastavro
from tqdm import tqdm
from tkinter.filedialog import askdirectory

def process_polar(folder):

    """
    Processes Polar VeritySense PPG and ACC files within a user-selected folder, converting them from CSV Avro format.
    """

    subfolders = [x for x in os.listdir(folder) if not x.startswith(".")] # filter hidden files

    # Create output directories
    if os.path.exists(folder + "/AVRO"):
        subfolders.remove("AVRO")

    if not os.path.exists(folder + "/AVRO"):
        os.mkdir(folder + "/AVRO")

    if not os.path.exists(folder + "/AVRO/ppg"):
        os.mkdir(folder + "/AVRO/ppg")

    if not os.path.exists(folder + "/AVRO/acc"):
        os.mkdir(folder + "/AVRO/acc")

    fields_ppg = [
        {"name": "timestamp", "type": "int"},
        {"name": "ppg1", "type": "float"},
        {"name": "ppg2", "type": "float"},
        {"name": "ppg3", "type": "float"},
        {"name": "ambient", "type": "float"},
    ]

    fields_acc = [
        {"name": "timestamp", "type": "int"},
        {"name": "acc_x", "type": "float"},
        {"name": "acc_y", "type": "float"},
        {"name": "acc_z", "type": "float"},
    ]

    for i, subf in enumerate(subfolders):

        csv_path = folder + "/" + subf + "/vs/"
        if not os.path.exists(csv_path):
            print(f"Skipping non-existent folder: {csv_path}")
            continue
        files = sorted(os.listdir(csv_path))

        print("\n")

        for _, f in enumerate(tqdm(files, desc = "Sto processando i file nella cartella N° " + str(i + 1) + " / " + str(len(subfolders)))):
            #### PPG ####
            if f.startswith("ppg"):
                ppg_df = pd.read_csv(csv_path + f)
                ppg_df.columns = ["timestamp", "ppg1", "ppg2", "ppg3", "ambient"]

                # Define the Avro schema
                schema = {
                    "type": "record",
                    "name": "PPG",
                    "fields": fields_ppg,
                }

                # Convert DataFrame to records
                records = ppg_df.to_dict(orient="records")

                # Write the records to an Avro file
                out_path_ppg = folder + "/AVRO/ppg/" + f.split(".csv")[0] + ".avro"
                if not os.path.exists(out_path_ppg):
                    with open(out_path_ppg, "wb") as out:
                            fastavro.writer(out, schema, records)

            #### ACC ####
            if f.startswith("acc"):
                acc_df = pd.read_csv(csv_path + f)
                acc_df.columns = ["timestamp", "acc_x", "acc_y", "acc_z"]

                # Define the Avro schema
                schema = {
                    "type": "record",
                    "name": "ACC",
                    "fields": fields_acc,
                }

                # Convert DataFrame to records
                records = acc_df.to_dict(orient="records")

                # Write the records to an Avro file
                out_path_acc = folder + "/AVRO/acc/" + f.split(".csv")[0] + ".avro"
                if not os.path.exists(out_path_acc):
                    with open(out_path_acc, "wb") as out:
                            fastavro.writer(out, schema, records)

def main():
    """
    Main function for interactive folder selection and processing.
    """

    # Ask user for the input directory
    print("Select the input folder:")
    input_dir = askdirectory(title="Select Input Folder")
    if not input_dir:
        print("No folder selected. Exiting.")
        return

    # Call the processing function
    process_polar(input_dir)
    print("\n\nConversione completata! Caricare ora la cartella 'AVRO' su Azure.\n\n")

if __name__ == "__main__":
    main()