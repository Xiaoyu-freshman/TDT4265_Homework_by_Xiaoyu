import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    width_predict=prediction_box[2]-prediction_box[0]
    height_predict=prediction_box[3]-prediction_box[1]
    width_gt=gt_box[2]-gt_box[0]
    height_gt=gt_box[3]-gt_box[1]
    #Calculate the edge of the overlap
    width_overlap= min(prediction_box[0],prediction_box[2],gt_box[0],gt_box[2])+width_predict+width_gt-max(prediction_box[0],prediction_box[2],gt_box[0],gt_box[2])
    height_overlap=min(prediction_box[1],prediction_box[3],gt_box[1],gt_box[3])+height_predict+height_gt-max(prediction_box[1],prediction_box[3],gt_box[1],gt_box[3])
    #Calculate the IoU
    if width_overlap <= 0 or height_overlap <= 0:
        iou = 0
    else:
        iou=width_overlap*height_overlap/(width_predict*height_predict+width_gt*height_gt-width_overlap*height_overlap)
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp/(num_tp+num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    else:
        return num_tp/(num_tp+num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    
    match_matrix=np.zeros([prediction_boxes.shape[0],gt_boxes.shape[0]])
    [row,column]=match_matrix.shape
    for i in range(row):
        for j in range(column):
            iou=calculate_iou(prediction_boxes[i,:],gt_boxes[j,:])
            if iou>= iou_threshold:
                match_matrix[i,j]=iou
    # Sort all matches on IoU in descending order
    match_row=[]
    match_column=[]
    #print('match_matrix',match_matrix)
    if match_matrix.size==1 and match_matrix>0:
        return prediction_boxes, gt_boxes
    elif match_matrix.size==0:
        return np.array([]), np.array([])
    else:
        while np.sum(match_matrix)>0:
            [index_r,index_c]=np.where(match_matrix==np.max(match_matrix))
            index_r= index_r[0]
            index_c=index_c[0]
            match_row.append(index_r)
            match_column.append(index_c)
            match_matrix[index_r,:]=0
            match_matrix[:,index_c]=0
                
    # Find all matches with the highest IoU threshold
        match_row=np.array(match_row)
        match_column=np.array(match_column)
        if match_row.size==0:
            return np.array([]), np.array([])
        else:
            return prediction_boxes[match_row,:], gt_boxes[match_column,:]
    


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    match_prediction, match_gt = get_all_box_matches(prediction_boxes,gt_boxes,iou_threshold)
    TP=match_prediction.shape[0]
    FP=prediction_boxes.shape[0]-TP
    FN=gt_boxes.shape[0]-TP
    return {"true_pos":TP, "false_pos": FP, "false_neg": FN}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    precision=0;recall=0
    for i in range(len(all_prediction_boxes)):
        res=calculate_individual_image_result(all_prediction_boxes[i],all_gt_boxes[i],iou_threshold)
        TP= res["true_pos"];FP=res["false_pos"];FN=res["false_neg"];
        precision=precision+calculate_precision(TP,FP,FN)
        recall=recall+calculate_recall(TP,FP,FN)
    return precision/len(all_prediction_boxes), recall/len(all_prediction_boxes)
    


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = [] 
    recalls = []
    for i in confidence_thresholds:
        satisfied_prediction_boxes=[]
        for j in range(len(confidence_scores)):
            tmp=np.where(confidence_scores[j]>=i)
            satisfied_prediction_boxes.append(all_prediction_boxes[j][tmp])
        precision,recall=calculate_precision_recall_all_images(satisfied_prediction_boxes, all_gt_boxes, iou_threshold)
        precisions.append(precision);recalls.append(recall)
        
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    precision_levels=np.zeros(recall_levels.shape)
    #Re-rank recalls and precisions from small to large
    precisions=precisions[np.argsort(recalls)]
    recalls=recalls[np.argsort(recalls)]
    for i in range(recall_levels.size):
        tmp=np.where(recalls>=recall_levels[i])[0]
        if tmp.size>0:
            precision_levels[i]=precisions[tmp[0]]
    average_precision = np.sum(precision_levels)/11
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
