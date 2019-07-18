import json
import pandas as pd 
import time

"""
需要一下文件： 
1、预测的json：bbox_level{}_test_results.json
2、test集的json：test.json
3、sample_submission.csv
"""


LABLE_LEVEL = 4
SCORE_THRESHOLD = 0.001


def json_to_dict(json_file_dir):
    with open(json_file_dir, "r") as json_file:
        json_dict = json.load(json_file)
        json_file.close()
    return json_dict

def get_threshold_result_list(label_level=LABLE_LEVEL, score_threshold=SCORE_THRESHOLD):
	detect_result_list = json_to_dict('bbox_level{}_test_results.json'.format(label_level))    # detect_result_list 是一个list # {'image_id': 2020005391, 'category_id': 43, 'bbox': [150.59866333007812, 332.810791015625, 370.6794128417969, 480.145263671875], 'score': 0.007447981275618076}
	result_Threshold_list = []                                                                      
	for result in detect_result_list:
		if result['score'] > score_threshold:
			result_Threshold_list.append(result)
	print("There are {} bboxes".format(len(result_Threshold_list)))
	return result_Threshold_list

def get_images_categories_info(label_level=LABLE_LEVEL):
	image_name_id_dict = {}
	image_id_name_dict = {}
	image_id_WH_dict = {}
	original_id_dict = {}
	id_original_dict = {}

	images_and_categories_dict = json_to_dict('test.json')
	images = images_and_categories_dict['images']
	categories = images_and_categories_dict['categories']
	for i in images:
		image_name_id_dict[i['file_name']] = i['id']
		image_id_name_dict[i['id']] = i['file_name']
		image_id_WH_dict[i['id']] = [i["width"], i["height"]]   # 是一个list
	for i in categories:
		original_id_dict[i["original_id"]] = i['id']
		id_original_dict[i['id']] = i['original_id']
	return image_name_id_dict, image_id_name_dict, image_id_WH_dict, original_id_dict, id_original_dict


def write_jsonresult_to_csv():
	ImageId = []
	PredictionString = []
	result_Threshold_list = get_threshold_result_list(LABLE_LEVEL, SCORE_THRESHOLD)
	_, image_id_name_dict, image_id_WH_dict, _, id_original_dict  = get_images_categories_info(label_level=LABLE_LEVEL)
	for bbox in result_Threshold_list:
		image_id = bbox['image_id']
		ImageId.append(image_id_name_dict[image_id][:-4])
		image_W = image_id_WH_dict[image_id][0]
		image_H = image_id_WH_dict[image_id][1]
		bbox_xmin = bbox['bbox'][0]/image_W
		bbox_ymin = bbox['bbox'][1]/image_H
		bbox_xmax = (bbox['bbox'][0] + bbox['bbox'][2])/image_W
		bbox_ymax = (bbox['bbox'][1] + bbox['bbox'][3])/image_H
		original_label = id_original_dict[bbox['category_id']]
		Confidence  = bbox['score']
		predictionstring = original_label + ' ' + str(Confidence) + ' ' + str(bbox_xmin)+ ' ' + str(bbox_ymin)+ ' ' + str(bbox_xmax)+ ' ' + str(bbox_ymax) + ' '
		PredictionString.append(predictionstring)
	
	print(len(ImageId))
	print(len(PredictionString))

	sample_csv = pd.read_csv('sample_submission.csv')
	sample_csv["PredictionString"] = ""
	# sample_ImageId = sample_csv["ImageId"].values.tolist()
	
	series_imageid = pd.Series(ImageId)
	series_predictionstring = pd.Series(PredictionString)
	# 写入并按图片名合并结果
	data = {'ImageId':series_imageid, "PredictionString":series_predictionstring}
	df = pd.DataFrame(data)
	df = pd.concat([sample_csv, df], ignore_index=True)
	df = df.groupby(by="ImageId")['PredictionString'].sum()
	series_imageid = df.index.tolist()
	series_imageid = pd.Series(series_imageid)
	series_predictionstring = pd.Series(df.values.tolist())
	data = {'ImageId':series_imageid, "PredictionString":series_predictionstring}
	df = pd.DataFrame(data)
	df.to_csv("level{}_{}_submission.csv".format(LABLE_LEVEL, SCORE_THRESHOLD), index=False)

if __name__ == "__main__":
	write_jsonresult_to_csv()






