# IMPORTS
import xml.etree.ElementTree as ET
import re
# CONSTS
WEB1_XML_PATH = "./web1.xml"
ALL_WEB_PATHS = ["./web1.xml", "./web2.xml","./web3.xml","./web4.xml"]
# CLASSES

# Function to read draw.io diagram and extract connection matrix
import xml.etree.ElementTree as ET
import re

class DrawIOObject:
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.object_types = {
            'thin_client': 1,
            'firewall': 2,
            'xenapp_server': 3,
            'cache_server': 4,
            'cellphone': 5,
            'xenclient_synchronizer': 6,
            'chassis': 7,
            'unknown': 8  # Default object type for unrecognized objects
        }
        self.connection_matrix, self.flattened_matrix = self.read_drawio_diagram()

    def extract_object_type(self, shape_name):
        if 'thin_client' in shape_name:
            return self.object_types['thin_client']
        elif 'firewall' in shape_name:
            return self.object_types['firewall']
        elif 'xenapp_server' in shape_name:
            return self.object_types['xenapp_server']
        elif 'cache_server' in shape_name:
            return self.object_types['cache_server']
        elif 'cellphone' in shape_name:
            return self.object_types['cellphone']
        elif 'xenclient_synchronizer' in shape_name:
            return self.object_types['xenclient_synchronizer']
        elif 'chassis' in shape_name:
            return self.object_types['chassis']
        else:
            return None

    def read_drawio_diagram(self):
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        objects = {}
        matrix_length = len(self.object_types)
        matrix = [[0] * matrix_length for _ in range(matrix_length)]

        for i in range(matrix_length):
            for j in range(matrix_length):
                matrix[i][j] = 0

        for cell in root.findall(".//mxCell"):
            cell_id = cell.get('id')
            cell_style = cell.get('style')
            cell_parent = cell.get('parent')

            if cell_id not in objects:
                objects[cell_id] = {'style': cell_style, 'parent': cell_parent}

        for conn in root.findall(".//mxCell[@edge='1']"):
            source_id = conn.get('source')
            target_id = conn.get('target')

            if source_id in objects and target_id in objects:
                source_obj_type = self.extract_object_type(objects[source_id]['style'])
                target_obj_type = self.extract_object_type(objects[target_id]['style'])
                if source_obj_type is not None and target_obj_type is not None:
                    matrix[source_obj_type-1][target_obj_type-1] = 1

        flattened_matrix = [value for sublist in matrix for value in sublist]

        return matrix, flattened_matrix
    
    def print_mstrix(self):     
        for name in self.object_types:
            print(name,end=",")
        print()
        i = 0
        values_list = list(self.object_types)
        for line in self.connection_matrix:
            print(values_list[i],end = ",")
            print(line)
            i +=1

# MAIN
def main():
    for path in ALL_WEB_PATHS:
        print("--------------------------------------------")
        web  =  DrawIOObject(path)
        print(path)
        web.print_mstrix()  

if __name__ == '__main__':
    main()
