from files import *

def play():
    pl.playsound("ring.mp3")

def send_mail_function(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    sender = 'sender@gmail.com'            # ENTER SENDER'S MAIL ID 
    recipient = "recipient@gmail.com"      # ENTER RECIPIENT'S MAIL ID 
    recipient = recipient.lower()
    msg['Subject'] = 'Violation'
    msg['From'] = sender
    msg['To'] = recipient
    text = MIMEText("A person is detected without wearing a mask.")
    msg.attach(text)
    img = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(img)
    server= smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(sender, 'password')         #ENTER PASSWORD OF SENDERS'S MAIL ID
    server.sendmail(sender, recipient, msg.as_string())
    print('sent to {}'.format(recipient))
    server.quit()



def detection():
    print("Starting....")
    r = 20
    w = 0
    h = 0
    path = "Model/coco.names"
    loc = open(path).read().strip().split("\n")
    np.random.seed(42)
    c = np.random.randint(0, 255, size=(len(loc), 3),dtype="uint8")
    weightm = "Model/yolov3.weights"
    confm = "Model/yolov3.cfg"

    layer = cv2.dnn.readNetFromDarknet(confm, weightm)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error in opening webcam...")
        exit()
    else:
        w = int (cap.get (cv2.CAP_PROP_FRAME_WIDTH))
        h = int (cap.get (cv2.CAP_PROP_FRAME_HEIGHT))
        dimen = (w, h)
        w = int (w * r / 100)
        h = int (h * r / 100)
        dimen = (w, h)




    b_dir=os.getcwd()
    b_dir=b_dir.replace('\\','/')

    data=b_dir+'/dataset'
    a=b_dir+'/Model'
    model=b_dir+'/Model/mask_detector.model'

    confidence=0.4


    caf=b_dir+'/face_detector/res10_300x300_ssd_iter_140000.caffemodel'

    print("Loading model...")
    prot = b_dir+'/face_detector/deploy.prototxt'
    weightm = caf
    face = cv2.dnn.readNet(prot, weightm)
    mask_model = load_model(model)
    print("Opening webcam...")
    video = VideoStream(src=0).start()

    
    def mask_prediction(capture, facet, mask):

        (h, w) = capture.shape[:2]
        img = cv2.dnn.blobFromImage(capture, 1.0, (300, 300),(104.0, 177.0, 123.0))
        wid=w
        hig=h
        facet.setInput(img)
        detect = facet.forward()
        det_face = []
        location = []
        predict =[]


        for i in range(0, detect.shape[2]):
            z = detect[0, 0, i, 2]
            if z > 0.5:
                b = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
                (X, Y, P, Q) = b.astype("int")
                (X, Y) = (max(0, X), max(0, Y))
                (P, Q) = (min(wid - 1, P), min(hig - 1, Q))
                f = capture[Y:Q, X:P]
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (224, 224))
                f = img_to_array(f)
                f = preprocess_input(f)

                det_face.append(f)
                location.append((X, Y, P, Q))

        if len(det_face) > 0:

            det_face = np.array(det_face, dtype="float32")
            predict = mask.predict(det_face, batch_size=32)

        return (location, predict)


    c=0
    while True:

        capture = video.read()
        capture = imutils.resize(capture, width=1200)

        resig = cv2.resize(capture, dimen, interpolation=cv2.INTER_AREA)

        (h,w)= capture.shape[:2]
        l_name=layer.getLayerNames()
        l_name= [l_name[i[0] - 1] for i in layer.getUnconnectedOutLayers()]
        img = cv2.dnn.blobFromImage(capture, 1 / 255.0, (224, 224), swapRB=True, crop=False)
        layer.setInput(img)
        start = time.time()
        op = layer.forward(l_name)
        end = time.time()

        store = []
        conf = []
        num = []

        for i in op:
            for j in i:
                score = j[5:]
                k = np.argmax(score)
                confi = score[k]
                if confi > 0.1 and k == 0:
                    b = j[0:4] * np.array([w, h, w, h])
                    (X,Y,wid,heg) = b.astype("int")
                    x = int(X - (wid / 2))
                    y = int(Y - (heg / 2))
                    store.append([x, y, int(wid), int(heg)])
                    conf.append(float(confi))
                    num.append(k)
        n=[]
        st=0
        if c % 3 == 0:
            q = []
            t = []

            st = cv2.dnn.NMSBoxes(store, conf, 0.5, 0.3)
            index = []
            for i in range(0, len(num)):
                if (num[i] == 0):
                    index.append(i)


            if len(st) > 0:
                for i in st.flatten():
                    (x, y) = (store[i][0], store[i][1])
                    (w, h) = (store[i][2], store[i][3])
                    q.append(x)
                    t.append(y)

            dt = []
            n=[]
            for i in range(0, len(q) - 1):
                for k in range(1, len(q)):
                    if (k == i):
                        break
                    else:
                        xd = (q[k] - q[i])
                        yd = (t[k] - t[i])
                        d = math.sqrt(xd*xd + yd*yd)
                        dt.append(d)
                        if (d <= 6912):
                            n.append(i)
                            n.append(k)
                        n = list(dict.fromkeys(n))


            color = (0, 0, 255)
            font=cv2.FONT_HERSHEY_SIMPLEX
            font1=cv2.FONT_HERSHEY_PLAIN
            for i in n:
                (x,y)= (store[i][0], store[i][1])
                (w,h)= (store[i][2], store[i][3])
                cv2.rectangle(capture, (x, y), (x + w, y + h), color, 2)
          
                cv2.putText(capture, "WARNING", (x, y - 5), font1, 0.5, color, 2)
            color = (0, 255, 0)
            if len(st) > 0:
                for i in st.flatten():
                    if (i in n):
                        break
                    else:
                        (x, y) = (store[i][0], store[i][1])
                        (w, h) = (store[i][2], store[i][3])
                        cv2.rectangle(capture, (x, y), (x + w, y + h), color, 2)
                       
                        cv2.putText(capture,"Ok", (x, y - 5), font1, 0.5, color, 2)

        text = "Social Distancing Violations: {}".format(len(n))

        cv2.putText(capture, text, (660, capture.shape[0] - 45),
                    font, 1, (0, 0, 255), 4)

        cv2.putText(capture, "Social Distancing And Mask Detector", (120,60),
                    font1, 3, (120, 175, 0), 4)


        
        high = "High Risk: " + str(len(n))

        safe = "Safe : " + str(len(st)-len(n))
        total = "Total : " + str(len(st))


        cv2.putText(capture, total, (10, 200),
                    font, 0.7, (47, 17, 8), 2)
        cv2.putText(capture, safe, (10, 165),
                    font, 0.7, (0, 255, 0), 2)
        cv2.putText(capture, high, (10, 130),
                    font, 0.7, (20,35,204), 2)


        cv2.rectangle(capture, (5,500), (400,100), (52, 50, 90), 2)
        cv2.putText(capture, "MASK DETECTION ANALYSIS :", (10,350),
                    font1, 1.5, (252, 90,6 ), 2)
        cv2.putText(capture, "-- NO MASK", (30, 380),
                    font1, 1.5, (35,13,251), 2)
        cv2.putText(capture, "-- MASK", (30, 420),
                    font1, 1.5, (72, 230, 51), 2)
        

        (location, predict) = mask_prediction(capture, face, mask_model)




        for (i, j) in zip(location, predict):

            (X, Y, P, Q) = i

            (mask, wmask) = j

            label = "Mask" if mask > wmask else "No Mask"
            color = (1, 255, 0) if label == "Mask" else (0, 0, 216)
            if label=="No Mask" :
                a=random.random()*1000
                play()
                save='IMAGES\img'+str(a)+'.jpg'
                cv2.imwrite(filename=save, img=capture)
                send_mail_function(save)


            label = "{}: {:.2f}%".format(label, max(mask, wmask) * 100)

            cv2.putText(capture, label, (X, Y - 10),
                font, 0.47, color, 2)
            cv2.rectangle(capture, (X, Y), (P, Q), color, 2)

        cv2.namedWindow('MASD', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('MASD', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MASD', capture)
        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            break



    cv2.destroyAllWindows()
    video.stop()

