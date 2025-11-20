# Menggunakan base image Python
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy file ke dalam Docker
COPY ./model ./model
COPY app.py .
COPY requirements.txt .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Expose port untuk Flask
EXPOSE 5000

# Perintah untuk menjalankan aplikasi Flask
CMD ["python", "app.py"]
