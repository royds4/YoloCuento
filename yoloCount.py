import subprocess
import openpyxl
import json
import os


def run():
    # Call the second Python script and capture its output
    folder_path = "C:/Users/Usuario/source/repos/2024-Livestream/data"
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        params = [
        '--video_file_path', folder_path+ '/'+file_name
        ]
        result = subprocess.run(['python', 'app.py'] + params, capture_output=True, text=True)
        output = result.stdout
        #variables = json.loads(output.strip())
        # buildExcel(variables)

   

def buildExcel(variables):
     # Create a new Excel workbook and select the active worksheet
    wb = openpyxl.Workbook()
    ws = wb.active

    # Populate the worksheet with the extracted variables
    row = 1
    col = 1
    # for data_labels in labels:
    #     ws.cell(row=row, column=data_labels.col, value=data_labels.Name)
    #     col+=1

    # ws.cell(row=row, column=col, value='INTERSECCIÓN')
    # ws.cell(row=row, column=col, value='DÍA')
    # ws.cell(row=row, column=col, value='FECHA')
    # ws.cell(row=row, column=col, value='HORA INICIO')
    # ws.cell(row=row, column=col, value='HORA FIN')
    # ws.cell(row=row, column=col, value='MOVIMIENTO')
    # ws.cell(row=row, column=col, value='AUTOS')
    # ws.cell(row=row, column=col, value='BUS')
    # ws.cell(row=row, column=col, value='SITP')
    # ws.cell(row=row, column=col, value='C2P')
    # ws.cell(row=row, column=col, value='C2G')
    # ws.cell(row=row, column=col, value='C3')
    # ws.cell(row=row, column=col, value='C4')
    # ws.cell(row=row, column=col, value='C5')
    # ws.cell(row=row, column=col, value='MOTO')


    #  Iterate over each dictionary in the list
    # for data_dict in variables:
    #     for key, value in data_dict.items():
    #     # Insert the value in the next row
    #         ws.cell(row=row, column=labels[key].col, value=value)
    #     row += 1  # Skip a row before processing the next dictionary

    # Save the workbook
    wb.save('CORREDOR VERDE CALLE 134 ATIP.xlsx')


if __name__ == "__main__":
    run()