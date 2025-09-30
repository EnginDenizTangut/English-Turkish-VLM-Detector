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
        
        # Renk eşleştirmesi - Türkçe renk isimlerini RGB değerlerine çevirir
        self.color_mapping = {
            'kırmızı': (0, 0, 255),      # BGR formatında
            'kirmizi': (0, 0, 255),
            'kırmızı': (0, 0, 255),
            'mavi': (255, 0, 0),
            'yeşil': (0, 255, 0),
            'yesil': (0, 255, 0),
            'sarı': (0, 255, 255),
            'sari': (0, 255, 255),
            'mor': (255, 0, 255),
            'turuncu': (0, 165, 255),
            'pembe': (203, 192, 255),
            'siyah': (0, 0, 0),
            'beyaz': (255, 255, 255),
            'gri': (128, 128, 128),
            'kahverengi': (42, 42, 165),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'lacivert': (139, 0, 0),
            'altın': (0, 215, 255),
            'altin': (0, 215, 255),
            'gümüş': (192, 192, 192),
            'gumus': (192, 192, 192),
            'default': (0, 255, 0)  # Varsayılan yeşil
        }
    
    def detect_objects(self, image_path):
        results = self.model(image_path)
        return results[0]
    
    def filter_objects_by_class(self, results, target_class):
        filtered_boxes = []
        filtered_confidences = []
        filtered_classes = []
        
        available_classes = list(self.class_names.values())
        
        # Renk bilgisini çıkar
        detected_color = self.extract_color_from_query(target_class)
        color_name = [name for name, value in self.color_mapping.items() if value == detected_color and name != 'default'][0]
        
        # Renk bilgisini target_class'dan çıkar
        clean_target = target_class.lower()
        for color in self.color_mapping.keys():
            if color != 'default' and color in clean_target:
                clean_target = clean_target.replace(color, '').strip()
                break
        
        llm_prompt = f"""
        Kullanıcı "{target_class}" nesnesini arıyor.
        
        Mevcut COCO sınıfları: {', '.join(available_classes)}
        
        Bu sınıflardan hangileri "{clean_target}" ile eşleşiyor? 
        
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
    
    def filter_objects_by_color(self, image_path, boxes, confidences, classes, target_color):
        """Renk bazında nesne filtreleme"""
        if target_color == self.color_mapping['default']:
            return boxes, confidences, classes
        
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return boxes, confidences, classes
            
            filtered_boxes = []
            filtered_confidences = []
            filtered_classes = []
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = map(int, box)
                
                # Bounding box içindeki alanı al
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # Renk analizi yap
                if self.is_object_color_match(roi, target_color):
                    filtered_boxes.append(box)
                    filtered_confidences.append(conf)
                    filtered_classes.append(cls)
            
            return filtered_boxes, filtered_confidences, filtered_classes
            
        except Exception as e:
            print(f"Renk filtreleme hatası: {e}")
            return boxes, confidences, classes
    
    def is_object_color_match(self, roi, target_color):
        """Nesnenin renginin hedef renkle eşleşip eşleşmediğini kontrol et"""
        try:
            # ROI'yi BGR'den RGB'ye çevir
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Hedef rengi BGR'den RGB'ye çevir
            target_rgb = target_color[::-1]  # BGR'den RGB'ye çevir
            
            # Renk mesafesi hesapla
            color_diff = np.sqrt(np.sum((rgb_roi - target_rgb) ** 2, axis=2))
            
            # Ortalama renk mesafesi
            avg_color_diff = np.mean(color_diff)
            
            # Renk eşleşme eşiği (0-255 arasında) - daha esnek
            color_threshold = 200
            
            # Eğer ortalama renk mesafesi eşikten küçükse eşleşme kabul et
            is_match = avg_color_diff < color_threshold
            
            print(f"Renk analizi: Hedef={target_rgb}, Ortalama mesafe={avg_color_diff:.1f}, Eşleşme={is_match}")
            
            return is_match
            
        except Exception as e:
            print(f"Renk eşleştirme hatası: {e}")
            return True  # Hata durumunda tüm nesneleri kabul et
    
    def extract_color_from_query(self, user_query):
        """Kullanıcı sorgusundan renk bilgisini çıkarır"""
        user_query_lower = user_query.lower()
        
        # Renk anahtar kelimelerini ara
        for color_name, color_value in self.color_mapping.items():
            if color_name in user_query_lower and color_name != 'default':
                return color_value
        
        # Eğer hiç renk bulunamazsa varsayılan yeşil döndür
        return self.color_mapping['default']
    
    def draw_detections(self, image_path, boxes, confidences, classes, output_path, color=None):
        image = cv2.imread(image_path)
        
        # Renk belirlenmemişse varsayılan yeşil kullan
        if color is None:
            color = self.color_mapping['default']
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = map(int, box)
            
            # Belirtilen renkte bounding box çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls}: {conf:.2f}"
            # Label'ı da aynı renkte yaz
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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
        
        # Kullanıcı sorgusundan renk bilgisini çıkar
        detected_color = self.extract_color_from_query(user_query)
        color_name = [name for name, value in self.color_mapping.items() if value == detected_color and name != 'default'][0]
        print(f"Tespit edilen renk: {color_name}")
        
        results = self.detect_objects(image_path)
        
        # Önce sınıf bazında filtrele
        boxes, confidences, classes = self.filter_objects_by_class(results, user_query)
        
        # Sonra renk bazında filtrele (eğer renk belirtilmişse)
        if detected_color != self.color_mapping['default']:
            print(f"Renk filtreleme uygulanıyor: {color_name}")
            boxes, confidences, classes = self.filter_objects_by_color(image_path, boxes, confidences, classes, detected_color)
        
        output_path = "output_detection.jpg"
        if boxes:
            self.draw_detections(image_path, boxes, confidences, classes, output_path, detected_color)
            print(f"Tespit edilen nesneler: {classes}")
            print(f"Sonuç görüntüsü kaydedildi: {output_path}")
            print(f"Bounding box rengi: {color_name}")
        else:
            print("Belirtilen nesneler bulunamadı.")
        
        return boxes, confidences, classes

def main():
    detector = VLMDetector()
    
    image_path = input("Görüntü dosyasının yolunu girin: ")
    user_query = input("Ne aramak istiyorsunuz? (örn: 'mavi arabaları göster', 'kırmızı kedileri bul'): ")
    
    detector.process_image(image_path, user_query)

if __name__ == "__main__":
    main()
