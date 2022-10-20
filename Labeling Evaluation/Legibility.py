###  Python script for the Code Editor in QGIS  ###
###################################################

## Title: Association.py
## Created: 14 August 2022
## Author: Rachid Oucheikh and Lars Harrie
## Description: Computes an legibility factor among the labels of landmarks and roads. 
## For definition of the association factor see the paper "Design, implementation and evaluation of generative deep learning models for map labeling".


import csv
import traceback
import sys

#Function to extract features existing inside an evaluation area from given layers 
def extract_labels(evaluationArea, Labels_layer):
    # Get a list of all the landmark labels created using one of the methods
    print(sys.path)
    labels_all = []
    for layer in Labels_layer:
        labels_all.append( [f for f in layer.getFeatures() ])
    labels_all = [ i for sub in labels_all for i in sub]
    print('The total number of all labels is :', len(labels_all))
    # Create a list with only those landmark features that are within the study area
    spatialIndexStructure_temp = QgsSpatialIndex()
    spatialIndexStructure_temp.addFeatures(labels_all)
    # Get the lines that overlap the label from the spatial index structure
    labelBB = evaluationArea.boundingBox()
    selected_features = spatialIndexStructure_temp.intersects(labelBB) 
    labels = []
    for feat in selected_features:
        #if (labels_all[feat].geometry().area() > 150):
        labels.append(labels_all[feat-1])
    print('Number of labels obtained in the evaluation area:', len(labels))

    return labels_all, labels


#update the metric results using the regional readability factor dictionary (regional = specific for an evaluation area). The best will be kept.
def update_dict(dict2, dict1):
    for key in dict1.keys():
        if key not in dict2.keys():
            dict2[key] = dict1[key]
        else:
            dict2[key] = min(dict1[key] , dict1[key])
    return dict2


def get_legibility_factor(spatialIndexStructure_leg, labels_all, labels, ID = 'ID'):
    legibility_factors = {}
    total_intersection = 0
    Sum_areas = 0
    i = j = 0
    for feature_i in labels:
        i +=1
        feature_i_name = feature_i[ID]
        label_area = feature_i.geometry().area()
        Sum_areas = Sum_areas  +  label_area
        intersection_i = 0
        labelBB = feature_i.geometry().boundingBox()
        selected_features = spatialIndexStructure_leg.intersects(labelBB) 
        for feat in selected_features:
            feature_j = labels_all[feat-1]
            feature_j_name = feature_j[ID]
            if feature_i!=feature_j :
                intersection = feature_i.geometry().intersection(feature_j.geometry())
                intersection_area = abs(intersection.area())
                if intersection_area > feature_i.geometry().area() * 0.9:
                    pass 
                else:
                	total_intersection += intersection_area
                	intersection_i += intersection_area
        if label_area!=0:
            pcnt = (intersection_i/label_area)
            if intersection_i==0:
            	feature_j_name = "None"
            if pcnt>1:
                pcnt = pcnt - round(pcnt)
            legibility_factors[feature_i_name] = [feature_j_name, label_area, intersection_i, pcnt]

    print("Total intersection", total_intersection, "total area : ", Sum_areas)
    return  legibility_factors


def get_legibility_factor_QGIS(spatialIndexStructure_leg, labels_Landmark_all, labels_Roads_all, labels_Roads, ID = 'ID'):
    legibility_factors = {}
    total_intersection = 0
    Sum_areas = 0
    i = 0
    for feature_i in labels_Roads:
        feature_i_name = feature_i[ID]
        label_area = feature_i.geometry().area()
        Sum_areas = Sum_areas  +  label_area
        intersection_i = 0
        labelBB = feature_i.geometry().boundingBox()
        selected_features = spatialIndexStructure_leg.intersects(labelBB) 
        for feat in selected_features:
            feature_j = labels_Landmark_all[feat-1]
            feature_j_name = feature_j[ID]
            intersection = feature_i.geometry().intersection(feature_j.geometry())
            intersection_area = abs(intersection.area())
            total_intersection += intersection_area
            intersection_i += intersection_area
        if label_area!=0:
            pcnt = (intersection_i/label_area)*100
            if intersection_i==0:
                feature_j_name = "None"
            if intersection_i!=0:
                i +=1
                print(intersection_i)


            legibility_factors[feature_i_name] = [feature_j_name, label_area, intersection_i, pcnt]

    print("Total intersection", total_intersection, "total area : ", Sum_areas)
    return  legibility_factors



root = QgsProject.instance().layerTreeRoot()
# Define the evaluation area (defined as a polygon in the QGIS environment)
layer_evaluation_areas = QgsProject.instance().mapLayersByName('Test_Area')
for feature in layer_evaluation_areas[0].getFeatures():
    if feature['ID'] == 0:
        feature_evaluation  = feature

evaluation_area = feature_evaluation.geometry()
# An extended evaluation area is used for the landmark buildings
evaluationAreaPolygon_extended = evaluation_area.buffer(400,1)




#********************************************* Computtation of the legibility factor for the QGIS labels **********************************************
output_file_QGIS = open('D:\\Code_QGIS\\Get_rasters\\legibility_QGIS.csv', 'w', newline='')
writer_QGIS = csv.writer(output_file_QGIS)
writer_QGIS.writerow(['Label ID'] + ['feature_j_name'] + ['Label area'] + ['Intersection area'] + ['Percentage'] )

def legibility_QGIS(output_file):
    # Create a list with only those QGIS labels that are within the study area
    #Load the landmark layer that contains the landmark labels
    layer_QGIS = QgsProject.instance().mapLayersByName('layer_QGIS_Roads_all')
    labels_Roads_all, labels_Roads = extract_labels(evaluation_area,layer_QGIS)
    print('Total number of labels obtained using QGIS-PAL for the test area:', len(labels_Roads_all))
    print('Number of labels obtained using QGIS-PAL for the test area:', len(labels_Roads))
    #QGIS_legibility = get_legibility_factor1(labels, ID='id')

    layer_QGIS = QgsProject.instance().mapLayersByName('QGIS_Label_Landmark')
    labels_Landmark_all, labels_Landmark = extract_labels(evaluation_area,layer_QGIS)
    print('Total number of labels obtained using QGIS-PAL for the test area:', len(labels_Landmark_all))
    print('Number of labels obtained using QGIS-PAL for the test area:', len(labels_Landmark))

    spatialIndexStructure_leg = QgsSpatialIndex()
    spatialIndexStructure_leg.addFeatures(labels_Landmark_all)

    QGIS_legibility = get_legibility_factor_QGIS(spatialIndexStructure_leg, labels_Landmark_all,  labels_Roads_all, labels_Roads, ID='id')

    for key in QGIS_legibility.keys():
        writer_QGIS.writerow([str(key)] + [str(QGIS_legibility[key][0])] + [str(QGIS_legibility[key][1])]  + [str(QGIS_legibility[key][2])] + [str(QGIS_legibility[key][3])])

    output_file.close()

    return QGIS_legibility

legibility_QGIS(output_file_QGIS)


#********************************************** Computation of legibility factor for Pix2pix labeling  ************************************************
output_file_Pix2pix = open('D:\\Code_QGIS\\Get_rasters\\legibility_Pix2pix.csv', 'w', newline='')
writer_Pix2pix = csv.writer(output_file_Pix2pix)
writer_Pix2pix.writerow(['Label ID'] + ['feature_j_name'] + ['Label area'] + ['Intersection area'] + ['Percentage'] )

def legibility_Pix2pix(output_file):
    # Create a list with only those manual labels that are within the study area
    # Associations of Pix2pix created labels for the first test set
    layer_Pix2pix_labels = QgsProject.instance().mapLayersByName('Final_Pix2pix_layer')  #  # The final labels obtained by Pix2Pix for both landmark and roads 
    labels_all, labels = extract_labels(evaluation_area, layer_Pix2pix_labels)
    pix2pix_legibility = {}
    labels_all = []
    for layer in layer_Pix2pix_labels:
        labels_all.append( [f for f in layer.getFeatures() ])
    labels_all = [ i for sub in labels_all for i in sub]

    print("number of labels : ", len(labels_all))


    spatialIndexStructure_Pix2pix = QgsSpatialIndex()
    spatialIndexStructure_Pix2pix.addFeatures(labels_all)

    pix2pix_legibility = get_legibility_factor(spatialIndexStructure_Pix2pix, labels_all, labels_all, ID = 'id')

    #Save the results to a csv file
    for key in pix2pix_legibility.keys():
        writer_Pix2pix.writerow([str(key)] + [str(pix2pix_legibility[key][0])] + [str(pix2pix_legibility[key][1])]  + [str(pix2pix_legibility[key][2])]  + [str(pix2pix_legibility[key][3])] )

    output_file.close()

    return pix2pix_legibility

legibility_Pix2pix(output_file_Pix2pix)

#********************************************** Computation of legibility factor for CycleGAN labeling  ************************************************
output_file_CycleGAN = open('D:\\Code_QGIS\\Get_rasters\\legibility_CycleGAN.csv', 'w', newline='')
writer_CycleGAN = csv.writer(output_file_CycleGAN)
writer_CycleGAN.writerow(['Label ID'] + ['feature_j_name'] + ['Label area'] + ['Intersection area'] + ['Percentage'] )

def legibility_CycleGAN(output_file):
    # Create a list with only those manual labels that are within the study area
    layer_cycleGAN_labels = QgsProject.instance().mapLayersByName('Final_CycleGAN_layer')  # The final labels obtained by CycleGAN for both landmark and roads 
    labels_all = []
    cycleGAN_legibility = {}
    for layer in layer_cycleGAN_labels:
        labels_all.append( [f for f in layer.getFeatures() ])
    labels_all = [ i for sub in labels_all for i in sub]

    print("number of labels : ", len(labels_all))


    spatialIndexStructure_CGAN = QgsSpatialIndex()
    spatialIndexStructure_CGAN.addFeatures(labels_all)

    cycleGAN_legibility = get_legibility_factor(spatialIndexStructure_CGAN, labels_all, labels_all, ID = 'id')

    #Save the results to a csv file
    for key in cycleGAN_legibility.keys():
        writer_CycleGAN.writerow([str(key)] + [str(cycleGAN_legibility[key][0])] + [str(cycleGAN_legibility[key][1])]  + [str(cycleGAN_legibility[key][2])]  + [str(cycleGAN_legibility[key][3])] )

    output_file.close()

    return legibility_CycleGAN
    
legibility_CycleGAN(output_file_CycleGAN)

#*************************************************  Computation of legibility factors for manual labeling ***********************************************

output_file_manual = open('D:\\Code_QGIS\\Get_rasters\\legibility_manual.csv', 'w', newline='')
writer_manual = csv.writer(output_file_manual)
writer_manual.writerow(['Label ID'] + ['feature_j_name'] + ['Label area'] + ['Intersection area'] + ['Percentage'] )

def legibility_manual(output_file):
    # Create a list with only those manual labels that are within the study area
    #Load the landmark layer that contains the landmark labels
    layer_manual_labels = QgsProject.instance().mapLayersByName('LMF_Landmark_Building_T')
    layer_manual_labels_roads = QgsProject.instance().mapLayersByName('L_ LMF_Road_Names_T')

    layer_manual_labels.extend(layer_manual_labels_roads)
    #Extract all the labels created manually and store them in a single list
    labels_manually_all = []
    for layer in layer_manual_labels:
        labels_manually_all.append( [f for f in layer.getFeatures() ])
    labels_manually_all = [ i for sub in labels_manually_all for i in sub]
    print('Number of manual labels in total:', len(labels_manually_all))

    labels_manually = []
    for feature_j in labels_manually_all:
        if feature_j.geometry().intersects(evaluation_area):
            labels_manually.append(feature_j)
    print('Number of manual labels in evaluation area:', len(labels_manually))

    spatialIndexStructure_leg = QgsSpatialIndex()
    spatialIndexStructure_leg.addFeatures(labels_manually_all)

    manual_legibility = get_legibility_factor(spatialIndexStructure_leg, labels_manually_all, labels_manually, ID='OBJECTID')


    #Save the results to a csv file
    for key in manual_legibility.keys():
        writer_manual.writerow([str(key)] + [str(manual_legibility[key][0])] + [str(manual_legibility[key][1])]  + [str(manual_legibility[key][2])] +[str(manual_legibility[key][3])] )

    output_file.close()

    return manual_legibility


legibility_manual(output_file_manual)
