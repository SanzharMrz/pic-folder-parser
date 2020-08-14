import cv2
import numpy as np


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


def score_photos(folder, target=None, create_copies=False, conf_thres=0.3, nms_thres=0.4):
    from shutil import copyfile
    import cv2
    import os
    import pandas as pd
    import pickle
    from sklearn.metrics import classification_report

    folder_last = list(filter(lambda x: len(x), folder.split('/')))[-1]
    output_folder = f'./results_{folder_last}'

    positive_folder = 'YES'
    negative_folder = 'NO'

    files_yes_path = os.path.join(output_folder, positive_folder)
    files_no_path = os.path.join(output_folder, negative_folder)

    if not os.path.exists(output_folder):
        os.makedirs(os.path.join(output_folder, positive_folder))
        os.makedirs(os.path.join(output_folder, negative_folder))

    model_weights = './model-weights/model.weights'
    model_cfg = './cfg/model.cfg'

    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    predicts = dict()
    files = os.listdir(folder)
    files_len = len(files)
    for i, file in enumerate(files, start=1):
        if not file.endswith(".jpg"):
            continue
        cap = cv2.VideoCapture(os.path.join(folder, file))
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(get_outputs_names(net))
            faces = post_process(frame, outs, conf_thres, nms_thres)
            print(f'{i} / {files_len} pictures processed, current {file} has {len(faces)} faces')
            if create_copies:
                output = os.path.join(folder, file)
                output_copy_path = os.path.join(files_yes_path, f"yes\\{file}") if len(faces) else os.path.join(
                    files_no_path, f"no\\{file}")
                copyfile(output, output_copy_path)
            predicts[file] = int(bool(faces))
        cap.release()
    print_string = f' and results saved in {output_folder}' if create_copies else ''
    print(f'All files processed' + print_string)
    if target:
        with open(target, "rb") as f:
            target_dict = pickle.load(f)
        metric_file = 'scores.csv'
        srt_trg = sorted(target_dict.items())
        srt_prd = sorted(predicts.items())
        assert [elem[0] for elem in srt_trg] == [elem[0] for elem in srt_prd]
        print('Classification report:')
        report = classification_report([elem[1] for elem in srt_trg], [elem[1] for elem in srt_prd], output_dict=True)
        print(classification_report([elem[1] for elem in srt_trg], [elem[1] for elem in srt_prd]))
        df = pd.DataFrame(report).transpose()
        print(f'Metrics for {folder} saved in {output_folder} as {metric_file}')
        df.to_csv(f'{os.path.join(output_folder, metric_file)}', index=False)
    return predicts
