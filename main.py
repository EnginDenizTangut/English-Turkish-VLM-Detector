import cv2
import numpy as np
from ultralytics import YOLO
import ollama
from PIL import Image
import io
import base64
import json

class VLMDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        
        self.class_names = self.model.names
    
    def detect_objects(self, image_path):
        results = self.model(image_path)
        return results[0]
    
    def filter_objects_by_class(self, results, target_class):
        filtered_boxes = []
        filtered_confidences = []
        filtered_classes = []
        
        available_classes = list(self.class_names.values())
        
        llm_prompt = f"""
        Kullanıcı "{target_class}" nesnesini arıyor.
        
        Mevcut COCO sınıfları: {', '.join(available_classes)}
        
        Bu sınıflardan hangileri "{target_class}" ile eşleşiyor? 
        
        Önemli: Türkçe kelimeleri İngilizce COCO sınıflarıyla eşleştir. Aynı anlama gelen farklı kelimeleri de düşün:
        
        İNSANLAR:
        - "insan", "kişi", "adam", "kadın", "çocuk", "bebek", "yaşlı", "genç" → "person"
        
        HAYVANLAR:
        - "kedi", "pisi", "miyav" → "cat"
        - "köpek", "it", "hav hav" → "dog" 
        - "kuş", "kanatlı" → "bird"
        - "at", "beygir" → "horse"
        - "inek", "sığır" → "cow"
        - "koyun", "kuzu" → "sheep"
        - "fil", "fildişi" → "elephant"
        - "ayı", "boz ayı" → "bear"
        - "zebra" → "zebra"
        - "zürafa", "uzun boyunlu" → "giraffe"
        - "hayvan", "canlı", "evcil hayvan" → "cat, dog, bird, horse, cow, sheep, elephant, bear, zebra, giraffe"
        
        ARAÇLAR:
        - "araba", "otomobil", "taşıt", "vasıta" → "car"
        - "kamyon", "tır", "yük aracı" → "truck"
        - "otobüs", "şehir otobüsü" → "bus"
        - "motosiklet", "moto", "motor" → "motorcycle"
        - "bisiklet", "velespit", "pedal" → "bicycle"
        - "uçak", "tayyare", "hava aracı" → "airplane"
        - "tren", "demiryolu" → "train"
        - "tekne", "gemi", "deniz aracı" → "boat"
        - "araç", "taşıt", "vasıta", "ulaşım aracı" → "car, truck, bus, motorcycle, bicycle, airplane, train, boat"
        
        YİYECEKLER:
        - "elma", "kırmızı elma", "yeşil elma" → "apple"
        - "muz", "sarı meyve" → "banana"
        - "pizza", "italyan yemeği" → "pizza"
        - "pasta", "kek", "tatlı" → "cake"
        - "donut", "halka tatlı" → "donut"
        - "sandviç", "ekmek arası" → "sandwich"
        - "portakal", "turunç" → "orange"
        - "brokoli", "yeşil sebze" → "broccoli"
        - "havuç", "turuncu sebze" → "carrot"
        - "sosisli", "hot dog" → "hot dog"
        - "yiyecek", "yemek", "besin", "gıda" → "apple, banana, pizza, cake, donut, sandwich, orange, broccoli, carrot, hot dog"
        
        EŞYALAR:
        - "sandalye", "oturak", "koltuk" → "chair"
        - "masa", "yemek masası", "çalışma masası" → "dining table"
        - "televizyon", "tv", "ekran" → "tv"
        - "laptop", "dizüstü", "bilgisayar" → "laptop"
        - "telefon", "cep telefonu", "mobil" → "cell phone"
        - "kitap", "yayın", "eser" → "book"
        - "saat", "zaman aleti" → "clock"
        - "çanta", "torba", "kese" → "handbag, backpack, suitcase"
        - "ayakkabı", "bot", "terlik" → "shoe, boot"
        - "giysi", "elbise", "kıyafet" → "clothing"
        
        ELEKTRONİK:
        - "elektronik", "teknoloji", "cihaz" → "tv, laptop, cell phone, remote, keyboard, mouse, microwave, oven, toaster, refrigerator"
        
        EV EŞYALARI:
        - "ev eşyası", "mobilya", "ev aleti" → "chair, couch, bed, dining table, toilet, tv, laptop, microwave, oven, toaster, sink, refrigerator"
        
        SADECE eşleşen COCO sınıf isimlerini virgülle ayırarak ver. Başka açıklama yapma.
        Örnek: person, car, truck
        """
        
        llm_response = self.ask_llm(llm_prompt)
        print(f"LLM sınıf eşleştirmesi: {llm_response}")
        
        lines = llm_response.strip().split('\n')
        matching_classes = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('"') and not line.startswith('Ayrıca'):
                if ',' in line:
                    classes = [cls.strip().lower() for cls in line.split(',')]
                    matching_classes.extend(classes)
                else:
                    if line.lower() in [name.lower() for name in available_classes]:
                        matching_classes.append(line.lower())
        
        if not matching_classes:
            matching_classes = [cls.strip().lower() for cls in llm_response.split(',')]
        
        print(f"Parse edilen sınıflar: {matching_classes}")
        
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(box.conf[0])
            
            if class_name.lower() in matching_classes:
                filtered_boxes.append(box.xyxy[0].cpu().numpy())
                filtered_confidences.append(confidence)
                filtered_classes.append(class_name)
        
        return filtered_boxes, filtered_confidences, filtered_classes
    
    def draw_detections(self, image_path, boxes, confidences, classes, output_path):
        image = cv2.imread(image_path)
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{cls}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image)
        return image
    
    def ask_llm(self, prompt):
        try:
            response = ollama.chat(model='llama3.1:latest', messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            return f"LLM hatası: {str(e)}"
    
    def process_image(self, image_path, user_query):
        print(f"Görüntü işleniyor: {image_path}")
        print(f"Kullanıcı sorgusu: {user_query}")
        
        results = self.detect_objects(image_path)
        
        boxes, confidences, classes = self.filter_objects_by_class(results, user_query)
        
        output_path = "output_detection.jpg"
        if boxes:
            self.draw_detections(image_path, boxes, confidences, classes, output_path)
            print(f"Tespit edilen nesneler: {classes}")
            print(f"Sonuç görüntüsü kaydedildi: {output_path}")
        else:
            print("Belirtilen nesneler bulunamadı.")
        
        return boxes, confidences, classes

def main():
    detector = VLMDetector()
    
    image_path = input("Görüntü dosyasının yolunu girin: ")
    user_query = input("Ne aramak istiyorsunuz? (örn: 'arabaları göster', 'kedileri bul'): ")
    
    detector.process_image(image_path, user_query)

if __name__ == "__main__":
    main()
