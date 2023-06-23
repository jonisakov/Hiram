# IMPORTS
import xml.etree.ElementTree as ET
import difflib
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# CONSTS
ALL_WEB_PATHS = ["./web1.xml", "./web2.xml","./web3.xml","./web4.xml"]
ALL_BACKUP_PATHS = ["./backup1.xml","./backup2.xml","./backup3.xml","./backup4.xml"]
ALL_MOBILE_PATHS = ["./mobile1.xml","./mobile2.xml","./mobile3.xml","./mobile4.xml"]
ALL_LOG_PATHS = ["./logging1.xml","./logging2.xml"]
TEST_PATHS = ["./test1.xml","./test2.xml", "./test3.xml"]
ALL_PATHS = ALL_WEB_PATHS + ALL_BACKUP_PATHS + ALL_LOG_PATHS + ALL_MOBILE_PATHS

# CLASSES

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
    
    def print_matrix(self):     
        for name in self.object_types:
            print(name,end=",")
        print()
        i = 0
        values_list = list(self.object_types)
        for line in self.connection_matrix:
            print(values_list[i],end = ",")
            print(line)
            i +=1


class FuzzyMatcher:
    def __init__(self, array):
        self.array = array

    def match(self, query):
        scores = 0
        ratio = difflib.SequenceMatcher(None, self.array, query).ratio()
        scores = ratio
        return scores



def perform_clustering(vectors):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(vectors)
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Reduce dimensionality to 2 dimensions using PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Plotting the results
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=cluster_labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clustering Results')

    # Add labels for each data point
    for i, vec in enumerate(reduced_vectors):
        plt.text(vec[0], vec[1], ALL_PATHS[i], fontsize=8, ha='center', va='center')

    plt.legend()
    plt.show()



def check_family_membership(vectors, test_vector, name_of_family):
 # Create labels for the vectors
    labels = np.zeros(len(vectors) + 1)
    labels[-1] = 1  # Assign label 1 to the test vector

    # Combine vectors and test_vector into a single array
    combined_vectors = np.vstack((vectors, test_vector))

    # Create a logistic regression classifier
    classifier = LogisticRegression()

    # Train the classifier
    classifier.fit(combined_vectors, labels)

    # Predict the label of the test_vector
    prediction = classifier.predict([test_vector])

    # Determine if the test_vector belongs to the same family
    if prediction == 1:
        print("The test vector belongs to the {0} family.".format(name_of_family))
    else:
        print("The test vector is not part of the {0} family.".format(name_of_family))


# MAIN
def main():
 
    # # PERFORM LEVINSTIEN DISTANCE CHECK AND PRINT NEREAST ANSWER
    # for test in TEST_PATHS:
    #     test_object = DrawIOObject(test)
    #     best_score = 0
    #     best_match = ''
    #     for path in ALL_PATHS:
    #         web  =  DrawIOObject(path)
    #         # web.print_mstrix()
    #         matcher = FuzzyMatcher(test_object.flattened_matrix)
    #         if best_score < matcher.match(web.flattened_matrix):
    #             best_match = path
    #             best_score = matcher.match(web.flattened_matrix)
    #     print("for test: {0} the best match was: {1}".format(test,best_match))

    
    # # PERFORM K-MEANS TO FIND THE "TYPE" OF THE ARCHITECTURE
    # all_vectors = []
    # for path in ALL_PATHS:
    #     web = DrawIOObject(path)
    #     all_vectors.append(web.flattened_matrix)
    # perform_clustering(all_vectors)
    
    # PERFORM A TEST FOR THE SIMALIRTY BETWEEEN THE FAMILY AND A TEST VECTOR
    for test in TEST_PATHS:
        print("===================================================================")
        all_vectors = []
        test_object = DrawIOObject(test)
        print("for the test object {0}".format(test))
        for path in ALL_BACKUP_PATHS:
            web = DrawIOObject(path)
            all_vectors.append(web.flattened_matrix)
        check_family_membership(all_vectors,test_object.flattened_matrix, "backup")
        all_vectors = []
        for path in ALL_LOG_PATHS:
            web = DrawIOObject(path)
            all_vectors.append(web.flattened_matrix)
        check_family_membership(all_vectors,test_object.flattened_matrix, "logging")
        all_vectors = []
        for path in ALL_WEB_PATHS:
            web = DrawIOObject(path)
            all_vectors.append(web.flattened_matrix)
        check_family_membership(all_vectors,test_object.flattened_matrix, "web")
        all_vectors = []
        for path in ALL_MOBILE_PATHS:
            web = DrawIOObject(path)
            all_vectors.append(web.flattened_matrix)
        check_family_membership(all_vectors,test_object.flattened_matrix, "mobile")




if __name__ == '__main__':
    main()