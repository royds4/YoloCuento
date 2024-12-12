import subprocess
import openpyxl
import json
import os

def run(folder_path):
    # Call the second Python script and capture its output
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        if 'dav' not in file_name:
            run(folder_path + '/' + file_name)
        else:
            params = [
            '--video_file_path', folder_path+ '/'+file_name
            ]
            result = subprocess.run(['venv/Scripts/python', 'app.py'] + params, capture_output=True, text=True)
            output = result.stdout
        variables = json.loads(output.strip())

        buildExcel(variables, file_name)

def buildExcel(variables, file_name):
    wb = openpyxl.Workbook()
    ws = wb.active
    startRow =2
    headers = set()
    for key,value in variables.items():
        headers.update(value.keys())
    headers = list(headers)
    
    for col_num, header in enumerate(headers, start=2):
        ws.cell(row=startRow, column=col_num, value=header)

    startRow += 1
    for property_name, value in variables.items():
        ws.cell(row=startRow, column=1, value=property_name)
        for col_num, header in enumerate(headers, start=2):
            ws.cell(row=startRow, column=col_num, value=value.get(header, ""))
        startRow += 1

    wb.save(f"{file_name}.xlsx")


if __name__ == "__main__":
    folderPath = "./data"
    run(folderPath)