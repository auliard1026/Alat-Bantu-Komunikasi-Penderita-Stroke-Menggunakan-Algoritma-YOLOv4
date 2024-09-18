import cv2
import time
import numpy as np
import telebot
import threading

bot = telebot.TeleBot('...') #isi dengan token bot

net1 = cv2.dnn.readNet("Model1", "Model2")
classes1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Switch', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

net2 = cv2.dnn.readNet("yolov4-custom.cfg", "yolov4-custom_best.weights")
classes2 = ['Adik', 'Bapak', 'Halo', 'Ibu', 'Kakek', 'Kamu', 'Makan', 'Minum', 'Nenek', 'Om', 'Pindah', 'Sama_sama', 'Saya', 'Switch', 'Tante', 'Terimakasih', 'Tidur', 'Toilet']

detection_messages = {
    'Makan': "Saya sangat lapar,tolong bantu saya untuk mengambil makan ",
    'Minum': "Saya merasa haus, tolong bantu saya untuk mengambil minum",
    'Toilet': "Saya perlu pergi ke toilet, tolong bantu saya ke toilet",
    'Pindah': "Saya ingin pindah posisi, tolong bantu saya untuk pindah posisi",
    'Tidur': "Saya merasa sangat mengantuk, saatnya untuk tidur."
}

current_net = net2
current_classes = classes2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

def set_cuda_preferences(net):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

set_cuda_preferences(current_net)

detected_objects = []
lock = threading.Lock()
detection_started = False  

message_log = []

def send_objects_to_telegram():
    global detected_objects
    while True:
        time.sleep(5)
        for i in range(5, 0, -1):
            print(f"Kirim pesan dalam: {i} detik")
            time.sleep(1)
        
        with lock:
            if detected_objects:
                for obj in detected_objects:
                    class_label = obj['class']
                    confidence = obj['confidence']
                    message = detection_messages.get(class_label, f"Pesan Terbaru {class_label}")
                    
                    if class_label in detection_messages:
                        bot.send_message('....', message)  #isi dengan bot id
                        message_log.append((time.strftime('%Y-%m-%d %H:%M:%S'), message))
                    
                    if class_label == 'Switch':
                        switch_model()

                detected_objects = []

def switch_model():
    global current_net, current_classes
    if current_net == net1:
        current_net = net2
        current_classes = classes2
        set_cuda_preferences(current_net) 
        bot.send_message('...', "Model switched to 2.") #isi dengan bot id
    else:
        current_net = net1
        current_classes = classes1
        set_cuda_preferences(current_net)  
        bot.send_message('...', "Model switched to 1.") #isi dengan bot id

def object_detection():
    global detection_started
    prev_frame_time = 0
    new_frame_time = 0

    confidence_threshold = 0.7
    nms_threshold = 0.4

    while True:
        if not detection_started:
            time.sleep(1)
            continue
        
        ret, frame = cap.read()

        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (256, 256), swapRB=True, crop=False)
        current_net.setInput(blob)
        output_layer_names = current_net.getUnconnectedOutLayersNames()
        outputs = current_net.forward(output_layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    if class_id < len(current_classes): 
                        class_label = current_classes[class_id]
                    else:
                        class_label = 'Unknown'
                        confidence = 0

                    if class_label in ignored_classes:
                        continue

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    min_dim = min(w, h)
                    new_x = int(center_x - min_dim / 2)
                    new_y = int(center_y - min_dim / 2)
                    new_w = new_h = min_dim

                    boxes.append([new_x, new_y, new_w, new_h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        with lock:
            detected_objects.clear()

            if len(indices) > 0: 
                for i in indices:
                    if isinstance(i, (list, tuple)):
                        i = i[0]  

                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]

                    if class_id < len(current_classes):
                        class_label = current_classes[class_id]
                    else:
                        class_label = 'Unknown'
                        confidence = 0

                    detected_objects.append({'class': class_label, 'confidence': confidence})

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    label = f"{class_label}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        cv2.imshow("Communication Engine", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@bot.message_handler(commands=['start'])
def handle_start(message):
    global detection_started
    detection_started = True
    bot.send_message(message.chat.id, "Sistem pendeteksian telah dimulai.")

@bot.message_handler(commands=['switch'])
def handle_switch(message):
    switch_model()
    bot.send_message(message.chat.id, "Model telah diganti.")

@bot.message_handler(commands=['check'])
def handle_check(message):
    global detection_started
    if detection_started:
        bot.send_message(message.chat.id, "Sistem pendeteksian sedang berjalan.")
    else:
        bot.send_message(message.chat.id, "Sistem pendeteksian belum dimulai atau terputus.")

@bot.message_handler(commands=['log'])
def handle_log(message):
    log_message = ""
    for log_entry in message_log:
        log_message += f"{log_entry[0]}: {log_entry[1]}\n"
    if log_message:
        bot.send_message(message.chat.id, log_message)
    else:
        bot.send_message(message.chat.id, "Belum ada pesan yang masuk.")

@bot.message_handler(commands=['menu'])
def handle_menu(message):
    menu = """
    Menu:
    1. Memulai Sistem - /start
    2. Lihat Log - /log
    3. Switch Model - /switch
    4. Check Koneksi - /check
    5. Bantuan - /help
    """
    bot.reply_to(message, menu)

@bot.message_handler(commands=['help'])
def handle_help(message):
    help_text = """
    Bantuan:
    - Gunakan perintah /start untuk memulai sistem.
    - Gunakan perintah /menu untuk melihat pilihan yang tersedia.
    - Gunakan perintah /check untuk melihat status koneksi sistem.
    - Gunakan perintah /log untuk melihat log deteksi.
    - Gunakan perintah /switch untuk beralih antara model deteksi objek.
    - Gunakan perintah /help untuk melihat bantuan ini.
    """
    bot.reply_to(message, help_text)

def send_startup_message():
    bot.send_message('....', "Sistem telah aktif. Gunakan perintah /menu untuk melihat menu.")  #isi dengan bot id

if __name__ == "__main__":
    send_startup_message()

    object_detection_thread = threading.Thread(target=object_detection)
    object_detection_thread.start()

    telegram_thread = threading.Thread(target=send_objects_to_telegram)
    telegram_thread.start()

    bot.polling()
