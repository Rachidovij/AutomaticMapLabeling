###  Python script for the Code Editor in QGIS  ###
###################################################

## Title: Association.py
## Created: 27 June 2022
## Author: Rachid Oucheikh and Lars Harrie
## Description: Computes an association factor between a landmark label and the landmark building object.
## For the definition of legibility, see the paper "Design, implementation and evaluation of generative deep learning models for map labeling".

import csv


path = "path to result folder"
# Open output file and write header
output_file = open(path + 'association_samples1.csv', 'w', newline='')
writer = csv.writer(output_file)
writer.writerow(['Landmark feature ID'] + ['Landmark feature area'] + ['Manual label ID'] + ['Text on label'] + ['Manual label area']  + ['Association factor manual'] +
['CycleGAN label area'] + ['Association factor CycleGAN'] )
#+ ['Pix2pix label area'] + ['Pix2pix factor']   + ['QGIS label area'] + ['Association factor QGIS'] )


#********************************************************************************* Funtions used in the program ****************************************************************
def extract_labels(evaluationArea, Labels_layer):
    # Get a list of all the landmark labels created using CycleGAN model
    labels_all = []
    for layer in Labels_layer:
        labels_all.append( [f for f in layer.getFeatures() ])
    labels_all = [ i for sub in labels_all for i in sub]
    # Create a list with only those landmark features that are within the study area
    labels = []
    for feature_j in labels_all:
        if (feature_j.geometry().intersects(evaluationArea)  and feature_j.geometry().area() > 150):
            labels.append(feature_j)

    return labels_all, labels

#update the association factor dictionary using the regional association factor dictionary (regional = specific for an evaluation area or a test set) 
def update_dict(dict1, dict2):
    for key in dict1.keys():
        if key not in dict2.keys():
            dict2[key] = dict1[key]
        else:
            dict2[key] = min(dict1[key] , dict1[key])
    return dict2


#If all the labels exist in the evaluation area (as it is the case for machine learning test data), then this function can be used
def Load_ML_Labels(Labels_layer):
    # Get a list of all the landmark labels created using CycleGAN model
    labels_all = []
    for layer in Labels_layer:
        labels_all.append( [f for f in layer.getFeatures() ])
    labels_all = [ i for sub in labels_all for i in sub]
    print('Number of labels obtained using CycleGAN in the evaluation area:', len(labels_all))

    return labels_all

#This function aims only to find the association between labels and features. The association factor is computed in the next function (get_association_factors).
def get_association_dict(labels, ID = 'ID'):
    # Create a dictionary that contain the linkage between landmark buildings and labels obtained using one of the labeliing techniques
    Associations_dict ={}

    # Go through all labels and compute the area of their intersections with the landmark features
    for feature_j in labels:
        feature_j_name = feature_j[ID]
        area = 0
        # Get the landmark features that overlap the label from the spatial index structure
        labelBB = feature_j.geometry().boundingBox()
        selected_features = spatialIndexStructure.intersects(labelBB)

        # Find the landmark feature that has the largest overlap with the label
        if len(selected_features) > 0:
            for feature_i_name in selected_features:
                old_area = area
                feature_i = features_landmark_all[feature_i_name - 1]  # Be aware of that the index refer to the list of all features
                intersection = feature_i.geometry().intersection(feature_j.geometry())
                area = intersection.area()
                if area >= old_area:
                    Associations_dict[feature_j_name] = feature_i_name

        # If no landmark feature overlaps the label then the closest landmark is selected
        else:
            labelCentroid = feature_j.geometry().centroid()
            closest_label_feature = spatialIndexStructure.nearestNeighbor(labelCentroid, 1, 10000000)
            Associations_dict[feature_j_name] = int(closest_label_feature[0])

    return Associations_dict



# Provide a list of association factors for each label type. We only do this for the labels
# that corresponds to the labels placed manually (which implies that some landmark buildings
# do not get any labels)
def get_association_factors(labels, Associations_dict, ID='ID'):
    association_results = {}
    duplicate = 0
    for feature_j in labels:    

        Label_ID = feature_j[ID]
        area_label  = feature_j.geometry().area()
        if area_label ==0:
            print('area_label=0')
            #area_label_cycleGAN  = 999 
            break    #If the area equals 0 then it should not be considered

        # Select the landmark features that overlap the label 
        labelBB = feature_j.geometry().boundingBox()
        selected_features = spatialIndexStructure.intersects(labelBB)

        association_temp = 0   # Temporary variable for computation of association factor

        # If there are landmark features overlapping the labels
        try:
            if len(selected_features) > 0:
                for feat_ID in selected_features:
                    feature_i = features_landmark_all[feat_ID - 1] # Be aware of that the index refer to the list of all features
                    area_feature = feature_i.geometry().area()
                    intersection = feature_i.geometry().intersection(feature_j.geometry())
                    area = intersection.area()
                    if feat_ID == Associations_dict[Label_ID]:
                        association_temp = association_temp + area
                        associatedFeature_ID = feat_ID
                    elif feat_ID in Associations_dict.values():
                        association_temp = association_temp - 2 * area
                    else:
                        association_temp = association_temp - area
            # If no landmark building is overlapping the landmark label
            else:
                association_temp = 0.0
                associatedFeature_ID = Associations_dict[Label_ID]
        except Exception as error:
            print(error):
            
        featureArea = features_landmark_all[associatedFeature_ID-1].geometry().area() 
        association_factor = association_temp / area_label
        label_text = features_landmark_all[associatedFeature_ID-1].attribute('name')
        #print('label_text', label_text)
        if associatedFeature_ID not in association_results.keys():
            association_results[associatedFeature_ID] =   [Label_ID, association_factor, label_text, featureArea, area_label ]
        elif association_factor > association_results[associatedFeature_ID][1]:
            existence = associatedFeature_ID in association_results.keys()
            duplicate += 1
            association_results[associatedFeature_ID] =   [Label_ID, association_factor, label_text, featureArea, area_label ] 
    print("number of duplicate keys is : ", duplicate)
    
    return association_results



# Define evaluation area (defined as a polygon in the QGIS environment)
layer_evaluation_areas = QgsProject.instance().mapLayersByName('Test_Area')
for feature in layer_evaluation_areas[0].getFeatures():
    if feature['id'] == 0:
        feature_evaluation  = feature
evaluation_area = feature_evaluation.geometry()
# An extended evaluation area is used for the landmark buildings
evaluationAreaPolygon_extended = evaluation_area.buffer(400,1)



# Get a list of all the landmark features (building objects)
layer_landmark = QgsProject.instance().mapLayersByName('Landmark')
features_landmark_all = []
for l in layer_landmark:
    features_landmark_all.append( [f for f in l.getFeatures() ])
features_landmark_all = [ i for sub in features_landmark_all for i in sub]
print('Number of landmark buildings in total:', len(features_landmark_all))


# Create a list with only those landmark features that are within the study area
features_landmark_study_area = []
for feature_j in features_landmark_all:
    if feature_j.geometry().intersects(evaluation_area):
        features_landmark_study_area.append(feature_j)
# Create a list with only those landmark features that are within an extended study area
features_landmark = []
for feature_j in features_landmark_all:
    if feature_j.geometry().intersects(evaluationAreaPolygon_extended):
        features_landmark.append(feature_j)
print('Number of landmark buildings in (extended) evaluation area:', len(features_landmark))

# Add the landmark to a spatial index data structure (R-tree)
spatialIndexStructure = QgsSpatialIndex()
spatialIndexStructure.addFeatures(features_landmark)



#**************************  Compute association factors for the manual labels  **********************************
# Get layer that contain minimum bounding boxes around all the labels.
layer_manual_labels = QgsProject.instance().mapLayersByName('LMF_Landmark_Building_T') # Manually defined labels

# Get a list of all the landmark labels created manually
labels_manually_all, labels_manually = extract_labels(evaluation_area,layer_manual_labels)
print('Number of manual labels in total:', len(labels_manually_all))
print('Number of manual labels in evaluation area:', len(labels_manually))

# Create a dictionary that contain the linkage between landmark buildings and manual labels
Associations_manual_dict = get_association_dict(labels_manually, ID='OBJECTID')
print('Number of key-value pairs in the dictionary for manual labels:', len(Associations_manual_dict.keys()))

#Compute association factors for the manual labels
Association_results_manual = get_association_factors(labels_manually, Associations_manual_dict, ID='OBJECTID')
   


#************************** Compute association factors for the labels created by CycleGAN  ***************************************

# Create a dictionary that contain the linkage between landmark buildings and the labels obtained by CycleGAN  for the first test set
layer_cycleGAN_labels_1 = QgsProject.instance().mapLayersByName('CycleGAN_labels_1_Final')  # Labels obtained by CycleGAN for test set 1
labels_all, labels = extract_labels(evaluation_area,layer_cycleGAN_labels_1)
print('Total number of labels obtained using CycleGAN for the first set:', len(labels_all))
print('Number of labels obtained using CycleGAN in the evaluation area for the first set:', len(labels))
Associations_cycleGan_dict_1 = get_association_dict(labels)
print('Number of key-value pairs in the dictionary for cycleGan labels:', len(Associations_cycleGan_dict_1.keys()))

#Compute association factors for the labels placed using CycleGAN for the first set
Association_results_cycleGan_1 = get_association_factors(labels, Associations_cycleGan_dict_1)

# Create a dictionary that contain the linkage between landmark buildings and the labels obtained by CycleGAN for the second test set
layer_cycleGAN_labels_2 = QgsProject.instance().mapLayersByName('CycleGAN_labels_2_Final')  # Labels obtained by CycleGAN for test set 2
labels_all, labels = extract_labels(evaluation_area, layer_cycleGAN_labels_2)
print('Total number of labels obtained using CycleGAN for the second set:', len(labels_all))
print('Number of labels obtained using CycleGAN in the evaluation area for the second set:', len(labels))
Associations_cycleGan_dict_2 = get_association_dict(labels)
print('Number of key-value pairs in the dictionary for cycleGan labels:', len(Associations_cycleGan_dict_2.keys()))

#Compute association factors for the labels placed using CycleGAN for the second set
Association_results_cycleGan_2 = get_association_factors(labels, Associations_cycleGan_dict_2)


dictionaries = [Association_results_cycleGan_1, Association_results_cycleGan_2]


# Create list of label IDs with the best association factors and use them in legibility factor 
list_ID = {'1': [], '2':[]}
Association_results_cycleGan ={}
for key in Association_results_manual.keys():
    if (key in Association_results_cycleGan_1.keys()) and (key in Association_results_cycleGan_2.keys()):
        if Association_results_cycleGan_1[key][1]> Association_results_cycleGan_2[key][1]:
            Association_results_cycleGan[key] = Association_results_cycleGan_1[key]
            list_ID['1'].append(Association_results_cycleGan_1[key][0])
        else: 
            Association_results_cycleGan[key] = Association_results_cycleGan_2[key]
            list_ID['2'].append(Association_results_cycleGan_2[key][0])       
    elif (key in Association_results_cycleGan_1.keys()):
        Association_results_cycleGan[key] = Association_results_cycleGan_1[key]
        list_ID['1'].append(Association_results_cycleGan_1[key][0])
    elif (key in Association_results_cycleGan_2.keys()):
        Association_results_cycleGan[key] = Association_results_cycleGan_2[key]
        list_ID['2'].append(Association_results_cycleGan_2[key][0]) 

list_csv = []
list_csv.append(list_ID['1'])
list_csv.append(list_ID['2'])
with open(path + 'Labels_for_Legibility_CycleGAN.csv', 'w')  as f_cycle:
    writer2 = csv.writer(f_cycle)
    writer2.writerows(list_csv)


#************************** Compute association factors for the labels created by Pix2pix  ***************************************

# Create a dictionary that contain the linkage between landmark buildings and the labels obtained by Pix2pix  for the first test set
layer_Pix2pix_labels_1 = QgsProject.instance().mapLayersByName('Pix2pix_labels_1_Final')  # Labels obtained by Pix2pix for test set 1
labels_all, labels = extract_labels(evaluation_area,layer_Pix2pix_labels_1)
print('Total number of labels obtained using Pix2pix for the first set:', len(labels_all))
print('Number of labels obtained using Pix2pix in the evaluation area for the first set:', len(labels))
Associations_Pix2pix_dict_1 = get_association_dict(labels, ID='id')
print('Number of key-value pairs in the dictionary for Pix2pix labels:', len(Associations_Pix2pix_dict_1.keys()))
Association_results_Pix2pix_1 = get_association_factors(labels, Associations_Pix2pix_dict_1, ID='id')

# Create a dictionary that contain the linkage between landmark buildings and the labels obtained by Pix2pix for the second test set
layer_Pix2pix_labels_2 = QgsProject.instance().mapLayersByName('CycleGAN_labels_2_Final')  # Labels obtained by Pix2pix for test set 2
labels_all, labels = extract_labels(evaluation_area, layer_Pix2pix_labels_2)
print('Total number of labels obtained using Pix2pix for the second set:', len(labels_all))
print('Number of labels obtained using Pix2pix in the evaluation area for the second set:', len(labels))
Associations_Pix2pix_dict_2 = get_association_dict(labels, ID='id')
print('Number of key-value pairs in the dictionary for Pix2pix labels:', len(Associations_Pix2pix_dict_2.keys()))
Association_results_Pix2pix_2 = get_association_factors(labels, Associations_Pix2pix_dict_2, ID='id')

# Create list of label IDs with the best association factors and use them in legibility factor 
list_ID = {'1': [], '2':[]}

Association_results_Pix2pix ={}
for key in Association_results_manual.keys():
    if (key in Association_results_Pix2pix_1.keys()) and (key in Association_results_Pix2pix_2.keys()):
        if Association_results_Pix2pix_1[key][1]  > Association_results_Pix2pix_2[key][1]:
            Association_results_Pix2pix[key] = Association_results_Pix2pix_1[key]
            list_ID['1'].append(Association_results_Pix2pix_1[key][0])
        else:
            Association_results_Pix2pix[key] = Association_results_Pix2pix_2[key]
            list_ID['2'].append(Association_results_Pix2pix_2[key][0])
    elif (key in Association_results_Pix2pix_1.keys()):
        Association_results_Pix2pix[key] = Association_results_Pix2pix_1[key]
        list_ID['1'].append(Association_results_Pix2pix_1[key][0])
    elif (key in Association_results_Pix2pix_2.keys()):
        Association_results_Pix2pix[key] = Association_results_Pix2pix_2[key]
        list_ID['2'].append(Association_results_Pix2pix_2[key][0])


# Save the considered labels from set 1 and set 2 in order to use them in legibility factor computation 
list_csv = []
list_csv.append(list_ID['1'])
list_csv.append(list_ID['2'])

with open(path + 'Labels_for_Legibility_Pix2Pix1.csv', 'w')  as f_pix:
    writer3 = csv.writer(f_pix)
    writer3.writerows(list_csv)


#***********************************************  QGIS created labels (optimisation method) *************************************************************************

# Go through all landmark objects which has associated ,anual labels and search if there is a QGIS label close to the
# centroid of the landmark.


Associations_QGIS_dict  ={}
Association_results_QGIS ={}

layer_QGIS = QgsProject.instance().mapLayersByName('layer_QGIS_all')
layer_QGIS = QgsProject.instance().mapLayersByName('Extra')
labels_all, labels = extract_labels(evaluation_area,layer_QGIS)
print('Total number of labels obtained using QGIS-PAL for the test area:', len(labels_all))
print('Number of labels obtained using QGIS-PAL for the test area:', len(labels))
Associations_QGIS_dict = get_association_dict(labels, ID='id')
print('Number of key-value pairs in the dictionary for QGIS-PAL labels:', len(Associations_QGIS_dict.keys()))

#Compute association factors for the QGIS labels
Association_results_QGIS = get_association_factors(labels, Associations_QGIS_dict, ID='id')


# *************************  Save the association result for all the methods ****************************************************

for key in Association_results_manual.keys():
    if key in Association_results_cycleGan.keys():
        cycle1, cycle2 = str(Association_results_cycleGan[key][4]) , str(Association_results_cycleGan[key][1])
    else:
        cycle1, cycle2 = "NAN", "NAN"
    if key in Association_results_Pix2pix.keys():
        pix1, pix2 = str(Association_results_Pix2pix[key][4]) , str(Association_results_Pix2pix[key][1])
    else:
        pix1, pix2 = "NAN", "NAN"
    if key in Association_results_QGIS.keys():
        qgis1, qgis2 = str(Association_results_QGIS[key][4]) , str(Association_results_QGIS[key][1])
    else:
        qgis1, qgis2 = "NAN", "NAN"
    writer.writerow([str(key)] + [str(Association_results_manual[key][3])] + [str(Association_results_manual[key][0])] + [str(Association_results_manual[key][2])] + 
        [str(Association_results_manual[key][4])] + [str(Association_results_manual[key][1])] +  [cycle1] + [cycle2] + [pix1] + [pix2] + [qgis1] + [qgis2])


output_file.close()

