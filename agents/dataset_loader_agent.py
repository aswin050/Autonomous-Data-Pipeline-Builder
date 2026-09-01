import pandas as pd
import os
import csv
import json
import sqlite3
import chardet
import warnings

warnings.filterwarnings("ignore")


class DatasetLoaderAgent:


    def __init__(self):

        self.report = {}



    # =====================================================
    # MAIN LOAD FUNCTION
    # =====================================================

    def load(self, file_path):


        print("="*60)
        print("DATASET LOADER AGENT")
        print("="*60)



        if not os.path.exists(file_path):

            print("Dataset file not found")

            return None



        try:


            self.report["file_name"] = (
                os.path.basename(file_path)
            )


            file_type = self.detect_format(file_path)


            print(
                "Detected format:",
                file_type
            )



            if file_type == "csv":

                df = self.load_csv(file_path)



            elif file_type == "json":

                df = self.load_json(file_path)



            elif file_type == "excel":

                df = self.load_excel(file_path)



            elif file_type == "parquet":

                df = pd.read_parquet(file_path)



            elif file_type == "xml":

                df = pd.read_xml(file_path)



            elif file_type == "sqlite":

                df = self.load_sqlite(file_path)



            elif file_type == "pickle":

                df = pd.read_pickle(file_path)



            elif file_type == "html":

                df = pd.read_html(file_path)[0]



            else:

                raise Exception(
                    "Unsupported format"
                )




            # Validation

            if df is None or df.empty:

                raise Exception(
                    "Dataset is empty"
                )



            if df.shape[1] == 1:

                print(
                    "⚠ Warning: Dataset contains only one column"
                )



            duplicate_columns = (
                df.columns[df.columns.duplicated()]
                .tolist()
            )


            if duplicate_columns:

                df = df.loc[
                    :,
                    ~df.columns.duplicated()
                ]



            memory = (
                df.memory_usage(deep=True)
                .sum()
                /
                1024
            )



            self.report.update({

                "format": file_type,

                "rows": df.shape[0],

                "columns": df.shape[1],

                "column_names":
                    df.columns.tolist(),

                "duplicate_columns":
                    duplicate_columns,

                "memory_usage_kb":
                    round(memory,2),

                "status":
                    "Success"

            })



            return df



        except Exception as e:


            print(
                "Loader Error:",
                e
            )


            self.report.update({

                "status":
                    "Failed",

                "error":
                    str(e)

            })


            return None





    # =====================================================
    # FORMAT DETECTION
    # =====================================================

    def detect_format(self,file_path):


        extension = (
            os.path.splitext(file_path)[1]
            .lower()
        )



        extension_map = {

            ".xlsx":"excel",
            ".xls":"excel",
            ".parquet":"parquet",
            ".xml":"xml",
            ".db":"sqlite",
            ".sqlite":"sqlite",
            ".pkl":"pickle",
            ".pickle":"pickle",
            ".html":"html",
            ".htm":"html"

        }



        if extension in extension_map:

            return extension_map[extension]



        # Content detection

        encoding = self.detect_encoding(
            file_path
        )



        with open(
            file_path,
            "r",
            encoding=encoding,
            errors="ignore"
        ) as f:

            sample = f.read(500)



        # JSON detection

        try:

            json.loads(sample)

            return "json"

        except:

            pass



        # XML detection

        if sample.strip().startswith("<"):

            return "xml"



        return "csv"






    # =====================================================
    # ENCODING DETECTION
    # =====================================================

    def detect_encoding(self,file_path):


        with open(
            file_path,
            "rb"
        ) as f:


            result = chardet.detect(
                f.read(200000)
            )


        encoding = result["encoding"]


        if encoding is None:

            encoding="utf-8"



        self.report[
            "encoding"
        ] = encoding



        return encoding





    # =====================================================
    # CSV LOADER
    # =====================================================

    def load_csv(self,file_path):


        encoding = self.detect_encoding(
            file_path
        )



        with open(
            file_path,
            "r",
            encoding=encoding,
            errors="ignore"
        ) as f:


            sample=f.read(10000)



        try:


            delimiter = csv.Sniffer().sniff(

                sample,

                delimiters=[
                    ",",
                    ";",
                    "\t",
                    "|"
                ]

            ).delimiter



        except:


            delimiter=","



        self.report[
            "separator"
        ] = delimiter



        print(
            "Detected separator:",
            delimiter
        )



        try:


            df=pd.read_csv(

                file_path,

                encoding=encoding,

                sep=delimiter,

                engine="python",

                on_bad_lines="skip"

            )


        except:


            df=pd.read_csv(

                file_path,

                encoding="latin1",

                engine="python",

                on_bad_lines="skip"

            )



        # JSON disguised as CSV

        if df.shape[1]==1:


            print(
                "Attempting recovery..."
            )


            try:


                df=self.load_json(
                    file_path
                )


            except:


                for sep in [
                    ",",
                    ";",
                    "\t",
                    "|"
                ]:


                    try:


                        temp=pd.read_csv(

                            file_path,

                            sep=sep,

                            encoding=encoding,

                            engine="python"

                        )


                        if temp.shape[1]>1:

                            df=temp

                            break


                    except:

                        pass



        return df





    # =====================================================
    # JSON LOADER
    # =====================================================

    def load_json(self,file_path):


        encoding=self.detect_encoding(
            file_path
        )


        with open(

            file_path,

            "r",

            encoding=encoding

        ) as f:


            data=json.load(f)




        if isinstance(data,list):

            return pd.json_normalize(data)



        if isinstance(data,dict):


            for key,value in data.items():


                if isinstance(value,list):

                    return pd.json_normalize(value)



            return pd.json_normalize(data)



        raise Exception(
            "Unsupported JSON structure"
        )





    # =====================================================
    # EXCEL LOADER
    # =====================================================

    def load_excel(self,file_path):


        sheets=pd.ExcelFile(
            file_path
        ).sheet_names



        print(
            "Available sheets:",
            sheets
        )


        return pd.read_excel(
            file_path,
            sheet_name=sheets[0]
        )





    # =====================================================
    # SQLITE LOADER
    # =====================================================

    def load_sqlite(self,file_path):


        conn=sqlite3.connect(
            file_path
        )


        tables=pd.read_sql(

            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """,

            conn

        )


        table=tables.iloc[0,0]


        df=pd.read_sql(

            f"SELECT * FROM {table}",

            conn

        )


        conn.close()


        return df





    # =====================================================
    # REPORT
    # =====================================================

    def get_report(self):

        return self.report