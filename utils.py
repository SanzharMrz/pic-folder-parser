import os
import pickle
from datetime import time
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def get_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second


def copy_to_results(folder, file, result, files_yes_path, files_no_path):
    output = os.path.join(folder, file)
    output_copy_path = os.path.join(files_yes_path if result else files_no_path, file)
    copyfile(output, output_copy_path)


def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    confidences = list()
    boxes = list()
    final_boxes = list()
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        h = i[0]
        box = boxes[h]
        final_boxes.append(box)
    return final_boxes


def get_faces(net, cap, conf_thres, nms_thres, upscale):
    faces_final = []
    while True:
        has_frame, frame_raw = cap.read()
        if not has_frame:
            break
        bounds = [None]
        if upscale:
            bounds += [
                [0, 0.5, 0, 0.5],
                [0.25, 0.75, 0, 0.5],
                [0.5, 1, 0, 0.5],

                [0, 0.5, 0.25, 0.75],
                [0.25, 0.75, 0.25, 0.75],
                [0.5, 1, 0.25, 0.75],

                [0, 0.5, 0.5, 1],
                [0.25, 0.75, 0.5, 1],
                [0.5, 1, 0.5, 1],
            ]
        for bound in bounds:
            if bound:
                frame = frame_raw[int(frame_raw.shape[0]*bound[0]):int(frame_raw.shape[0]*bound[1]), int(frame_raw.shape[1]*bound[2]):int(frame_raw.shape[1]*bound[3])]
            else:
                frame = frame_raw
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(get_outputs_names(net))
            faces = post_process(frame, outs, conf_thres, nms_thres)
            faces_final.extend(faces)
    cap.release()
    return faces_final


def score_photos(folder, target=None, output_folder=None, create_copies=False, conf_thres=0.3, nms_thres=0.4, dynamic_window=1, do_rename=True, upscale=False):
    positive_folder = 'yes'
    negative_folder = 'no'

    files_yes_path = os.path.join(output_folder, positive_folder)
    files_no_path = os.path.join(output_folder, negative_folder)

    if any([not os.path.exists(folder) for folder in [output_folder, files_yes_path, files_no_path]]):
        os.makedirs(files_yes_path)
        os.makedirs(files_no_path)

    model_weights = './model-weights/model.weights'
    model_cfg = './cfg/model.cfg'

    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    predicts = dict()
    files = sorted(os.listdir(folder))
    files_len = len(files)
    buffer = []
    for i, file in enumerate(files, start=1):
        name = ".".join(file.split(".")[:-1])
        extension = file.split(".")[-1]
        if name.endswith("_processed") or extension not in ['jpg', 'png']:
            continue

        if do_rename:
            os.rename(os.path.join(folder, file), os.path.join(folder, name + "_processed." + extension))
            file = name + "_processed." + extension

        ts = name[:8]
        cap_time = time(int(ts[:2]), int(ts[3:5]), int(ts[6:8]))

        if buffer and get_seconds(cap_time) - get_seconds(buffer[-1]['time']) >= 3:
            if create_copies:
                for cap in buffer:
                    if cap['processed']:
                        continue
                    copy_to_results(folder, cap['file'], len(cap['faces']) > 0, files_yes_path, files_no_path)
            buffer = []

        if not buffer or (get_seconds(cap_time) - get_seconds(buffer[-1]['time']) < 3):
            buffer.append(
                {
                    "name": name,
                    "extension": extension,
                    "file": file,
                    "time": cap_time,
                    "processed": False
                }
            )

        cap = cv2.VideoCapture(os.path.join(folder, file))
        faces = get_faces(net, cap, conf_thres, nms_thres, upscale)
        buffer[-1]["faces"] = faces
        print(f'{i} / {files_len} pictures processed, current {file} has {len(faces)} faces')
        predicts[file.replace("_processed", "")] = int(len(faces) > 0)

        if len(buffer) == 2 * dynamic_window + 1:
            has_faces = len(buffer[dynamic_window]['faces']) > 0
            if has_faces:
                no_faces_neighboors = [len(cap['faces']) == 0 for i, cap in enumerate(buffer) if i != dynamic_window]
                if sum(no_faces_neighboors) > dynamic_window:
                    has_faces = False
            else:
                has_faces_neighboors = [len(cap['faces']) > 0 for i, cap in enumerate(buffer) if i != dynamic_window]
                if sum(has_faces_neighboors) > dynamic_window:
                    has_faces = True
            predicts[buffer[dynamic_window]['file'].replace("_processed", "")] = has_faces

            if create_copies:
                copy_to_results(folder, buffer[dynamic_window]['file'], has_faces, files_yes_path, files_no_path)
                buffer[dynamic_window]['processed'] = True
                for cap in buffer[:dynamic_window]:
                    if cap['processed']:
                        continue
                    copy_to_results(folder, cap['file'], len(cap['faces']) > 0, files_yes_path, files_no_path)
                    cap['processed'] = True
            buffer = buffer[1:]

    for cap in buffer:
        if cap['processed']:
            continue
        copy_to_results(folder, cap['file'], len(cap['faces']) > 0, files_yes_path, files_no_path)

    print_string = f' and results saved in {output_folder}' if create_copies else ''
    print(f'All files processed' + print_string)
    if target:
        with open(target, "rb") as f:
            target_dict = pickle.load(f)
        metric_file = 'scores.csv'
        srt_trg, srt_prd = sorted(target_dict.items()), sorted(predicts.items())
        val_trg, val_prd = [elem[1] for elem in srt_trg],  [elem[1] for elem in srt_prd]
        assert [elem[0] for elem in srt_trg] == [elem[0] for elem in srt_prd]
        print('Classification report:')
        report = classification_report(val_trg, val_prd, output_dict=True)
        print(classification_report(val_trg, val_prd))
        df = pd.DataFrame(report).transpose()
        print(f'Metrics for {folder} saved in {output_folder} as {metric_file}')
        df.to_csv(f'{os.path.join(output_folder, metric_file)}', index=False)
    return predicts
