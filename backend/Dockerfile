# 1. اختيار صورة بايثون خفيفة ومستقرة
FROM python:3.10-slim

# 2. إعداد متغيرات البيئة لتحسين الأداء
# (يمنع بايثون من كتابة ملفات pyc ويجعل السجلات تظهر فوراً)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. تثبيت مكتبات النظام الضرورية (أهم خطوة لمشاريع الـ OCR)
# هذه المكتبات تحتاجها OpenCV و EasyOCR لتعمل على لينكس
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. تحديد مسار العمل داخل الحاوية
WORKDIR /code

# 5. نسخ ملف المتطلبات وتثبيته أولاً (للاستفادة من الكاشينج)
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. نسخ كود المشروع (مجلد app) إلى الحاوية
COPY ./app /code/app

# 7. إنشاء مجلد لموديلات EasyOCR لتجنب مشاكل الصلاحيات (اختياري لكن مفضل)
RUN mkdir -p /root/.EasyOCR

# 8. فتح المنفذ (Port) الذي سيعمل عليه التطبيق
EXPOSE 8000

# 9. أمر تشغيل السيرفر عند بدء الحاوية
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
