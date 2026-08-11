import os
import zipfile


class SmartDatasetValidator:


    def __init__(self):

        self.report = {}



    # =====================================================
    # MAIN VALIDATION FUNCTION
    # =====================================================

    def validate(self, file_path):


        print("="*60)
        print("="*60)



        if not os.path.exists(file_path):

            return self.fail(
                "File not found"
            )



        self.report["file"] = file_path



        size = (
            os.path.getsize(file_path)
            /
            (1024*1024)
        )


        self.report["file_size_mb"] = round(
            size,
            3
        )



        real_type = self.detect_real_type(
            file_path
        )


        self.report["detected_type"] = real_type


        print(
            "Detected:",
            real_type
        )



        if size == 0:

            return self.fail(
                "Empty file"
            )



        # =================================================
        # ZIP DATASET HANDLING
        # =================================================

        if real_type == "zip":


            extracted = self.extract_zip(
                file_path
            )


            if extracted is None:

                return self.fail(
                    "ZIP does not contain dataset"
                )



            file_path = extracted


            self.report[
                "extracted_file"
            ] = extracted



            real_type = self.detect_real_type(
                file_path
            )



            self.report[
                "detected_type_after_extract"
            ] = real_type



        # =================================================
        # NOTEBOOK DETECTION
        # =================================================

        if real_type == "jupyter":


            return self.fail(
                "Jupyter Notebook detected. Upload dataset file."
            )



        # =================================================
        # SUPPORTED DATASETS
        # =================================================

        supported_formats = [

            "csv",
            "json",
            "excel",
            "parquet",
            "xml",
            "sqlite"

        ]



        if real_type in supported_formats:


            print(
                "✔ Dataset validation successful"
            )


            self.report["status"] = "Valid"


            return file_path, self.report




        return self.fail(
            "Unsupported file format"
        )





    # =====================================================
    # FILE TYPE DETECTION
    # =====================================================

    def detect_real_type(self,file_path):


        with open(
            file_path,
            "rb"
        ) as f:

            signature = f.read(10)



        # ZIP / XLSX

        if signature.startswith(
            b"PK"
        ):


            try:

                with zipfile.ZipFile(
                    file_path
                ) as z:


                    names = z.namelist()



                    if any(
                        "notebook" in x.lower()
                        for x in names
                    ):

                        return "jupyter"



                    return "zip"



            except:

                return "zip"




        # SQLite

        if signature.startswith(
            b"SQLite"
        ):

            return "sqlite"



        # Parquet

        if signature.startswith(
            b"PAR1"
        ):

            return "parquet"




        ext = (
            os.path.splitext(file_path)[1]
            .lower()
        )



        extension_map = {


            ".csv":"csv",

            ".json":"json",

            ".xlsx":"excel",

            ".xls":"excel",

            ".xml":"xml",

            ".parquet":"parquet",

            ".db":"sqlite",

            ".sqlite":"sqlite",

            ".ipynb":"jupyter"


        }



        return extension_map.get(
            ext,
            "unknown"
        )






    # =====================================================
    # ZIP EXTRACTION
    # =====================================================

    def extract_zip(self,zip_path):


        print(
            "ZIP dataset detected"
        )



        folder = os.path.dirname(
            zip_path
        )



        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:


            zip_ref.extractall(
                folder
            )



        datasets = []



        for root,dirs,files in os.walk(folder):


            for file in files:


                if file.lower().endswith(

                    (
                    ".csv",
                    ".json",
                    ".xlsx",
                    ".parquet"
                    )

                ):


                    datasets.append(

                        os.path.join(
                            root,
                            file
                        )

                    )



        if len(datasets)==0:


            print(
                "❌ No dataset found"
            )


            return None




        print("\nDatasets found:")



        for i,file in enumerate(datasets):


            print(
                f"{i+1}. {file}"
            )



        # Prefer train dataset

        for file in datasets:


            if "train" in file.lower():


                print(
                    "\nSelected:",
                    file
                )


                return file




        print(
            "\nSelected:",
            datasets[0]
        )


        return datasets[0]






    # =====================================================
    # FAILURE REPORT
    # =====================================================

    def fail(self,message):


        print(
            "❌",
            message
        )



        self.report.update({

            "status":"Rejected",

            "reason":message

        })



        return None, self.report






    # =====================================================
    # REPORT
    # =====================================================

    def get_report(self):

        return self.report