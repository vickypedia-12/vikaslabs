"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink, Github } from "lucide-react";

const experiments = [
  {
    id: "api-rate-limiter", // Keeping the same ID
    name: "Website Route Scraper",
    description:
      "A Scrapy-based crawler to scrape a website and recursively extract all internal routes (links).",
    preview: `import scrapy
from urllib.parse import urljoin

class RouteSpider(scrapy.Spider):
    name = "route_spider"
    start_urls = ["https://example.com"]

    def parse(self, response):
        # Extract all internal links
        links = response.css('a::attr(href)').getall()
        for link in links:
            full_url = urljoin(response.url, link)
            if full_url.startswith(response.url):  # Only crawl internal routes
                yield {"url": full_url}
                yield scrapy.Request(full_url, callback=self.parse)`,
    fullCode: `import scrapy
from urllib.parse import urljoin

class RouteSpider(scrapy.Spider):
    name = "route_spider"

    # The domain to crawl (change as needed)
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com"]

    def parse(self, response):
        # Extract and normalize all links on the page
        links = response.css('a::attr(href)').getall()
        for link in links:
            full_url = urljoin(response.url, link)
            
            # Only follow links within the same domain
            if full_url.startswith("https://example.com"):
                yield {
                    "url": full_url
                }
                
                # Recursively follow the links (depth-first crawling)
                yield scrapy.Request(full_url, callback=self.parse)
                
    # Optional: To prevent revisiting the same URL multiple times
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited = set()

    def parse(self, response):
        if response.url in self.visited:
            return
        self.visited.add(response.url)

        links = response.css('a::attr(href)').getall()
        for link in links:
            full_url = urljoin(response.url, link)
            if full_url.startswith("https://example.com"):
                yield {"url": full_url}
                yield scrapy.Request(full_url, callback=self.parse)`,
    github: "https://github.com/username/website-route-scraper",
    demo: "https://demo.example.com/route-scraper",
  },
  {
    id: "data-pipeline", // Same ID
    name: "Logo Detection",
    description:
      "Deep learning based logo detection system using YOLOv5 to detect and localize logos in images.",
    preview: `# Example inference using YOLOv5
import torch
from pathlib import Path
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
img = Image.open('test.jpg')
results = model(img)
results.print()  # Show detected logos
results.save(save_dir=Path('output/'))`,
    fullCode: `import torch
from pathlib import Path
from PIL import Image, ImageDraw

class LogoDetector:
    def __init__(self, model_path: str = 'best.pt', device: str = 'cpu'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(device)
    
    def detect_and_draw(self, img_path: str, output_dir: str = 'output'):
        img = Image.open(img_path).convert('RGB')
        results = self.model(img)
        detections = results.xyxy[0]  # xmin, ymin, xmax, ymax, conf, cls
        
        draw = ImageDraw.Draw(img)
        for *box, conf, cls in detections.tolist():
            xmin, ymin, xmax, ymax = map(int, box)
            label = f"{self.model.names[int(cls)]}: {conf:.2f}"
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
            draw.text((xmin, ymin - 10), label, fill='red')
        
        Path(output_dir).mkdir(exist_ok=True)
        save_path = Path(output_dir) / Path(img_path).name
        img.save(save_path)
        return save_path

# Usage example
if __name__ == '__main__':
    detector = LogoDetector(model_path='best.pt')
    saved = detector.detect_and_draw('logo_test.jpg')
    print(f"Output image saved to: {saved}")`,
    github: "https://github.com/vickypedia-12/logo-Detection",
    demo: "https://demo.example.com/logo-detection",
  },
  {
    id: "ml-deployment", // Same ID
    name: "Food-101 Classifier Deployment",
    description:
      "FastAPI-based deployment of a Food-101 image classification model for predicting food categories with top-5 suggestions.",
    preview: `from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision import transforms
from PIL import Image

app = FastAPI(title="Food-101 Classifier", version="1.0.0")

# Load model
model = torch.load("food101_model.pth", map_location="cpu")
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load labels
with open("classes.txt") as f:
    class_labels = [line.strip() for line in f]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probabilities, 5)
        
        predictions = [
            {"label": class_labels[idx], "probability": float(prob)}
            for prob, idx in zip(top5_probs, top5_indices)
        ]
        return {"top_1": predictions[0], "top_5": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))`,
    fullCode: `from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
import torch
from torchvision import transforms
from PIL import Image
from typing import List, Dict
from datetime import datetime
import redis
import json
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('food101_requests_total', 'Total prediction requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('food101_request_duration_seconds', 'Request duration')

security = HTTPBearer()
app = FastAPI(title="Food-101 Classifier API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis cache
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Load Food-101 model
model = torch.load("food101_model.pth", map_location="cpu")
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class labels
with open("classes.txt") as f:
    class_labels = [line.strip() for line in f]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    token: str = Depends(verify_token)
):
    start_time = time.time()
    try:
        # Cache check
        cache_key = f"food101:{file.filename}"
        cached = redis_client.get(cache_key)
        if cached:
            logger.info("Returning cached prediction")
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="cached").inc()
            return json.loads(cached)

        # Preprocess image
        image = Image.open(file.file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probabilities, 5)

        predictions = [
            {"label": class_labels[idx], "probability": float(prob)}
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        response = {
            "timestamp": datetime.now().isoformat(),
            "top_1": predictions[0],
            "top_5": predictions
        }

        # Cache for 5 minutes
        redis_client.setex(cache_key, 300, json.dumps(response))
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
        REQUEST_DURATION.observe(time.time() - start_time)

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`,
    github: "https://github.com/vickypedia-12/foodle",
    demo: "https://demo.example.com/foodle",
  },
];

export default function ExperimentsSection() {
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(
    new Set()
  );

  const toggleExpanded = (projectId: string) => {
    setExpandedProjects((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(projectId)) {
        newSet.delete(projectId);
      } else {
        newSet.add(projectId);
      }
      return newSet;
    });
  };

  return (
    <section id="experiments" className="py-20 bg-background">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-foreground mb-4">
            My <span style={{ color: "hsl(217, 91%, 60%)" }}>Experiments</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Real-world AI/ML and API projects showcasing production-ready
            implementations with comprehensive code examples and deployment
            strategies.
          </p>
        </div>

        <div className="space-y-8">
          {experiments.map((experiment) => (
            <div
              key={experiment.id}
              className="bg-card border border-border rounded-lg p-8 hover:border-primary/50 transition-colors duration-200"
            >
              <div className="flex items-start justify-between mb-6">
                <div className="flex-1">
                  <h3 className="text-2xl font-semibold text-foreground mb-3">
                    {experiment.name}
                  </h3>
                  <p className="text-muted-foreground text-lg leading-relaxed">
                    {experiment.description}
                  </p>
                </div>
                <div className="flex items-center gap-3 ml-6">
                  <a
                    href={experiment.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 rounded-md bg-secondary hover:bg-secondary/80 transition-colors"
                  >
                    <Github className="h-5 w-5 text-foreground" />
                  </a>
                  <a
                    href={experiment.demo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 rounded-md bg-primary hover:bg-primary/90 transition-colors"
                  >
                    <ExternalLink className="h-5 w-5 text-primary-foreground" />
                  </a>
                </div>
              </div>

              <div className="bg-secondary rounded-lg border border-border overflow-hidden">
                <div className="flex items-center justify-between bg-secondary/50 px-4 py-3 border-b border-border">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-destructive"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <div className="w-3 h-3 rounded-full bg-primary"></div>
                  </div>
                  <span className="text-sm text-muted-foreground font-mono">
                    {experiment.name.toLowerCase().replace(/\s+/g, "-")}.py
                  </span>
                </div>

                <div className="p-4">
                  <pre className="text-sm overflow-x-auto">
                    <code className="font-mono text-foreground whitespace-pre-wrap">
                      {experiment.preview}
                    </code>
                  </pre>
                </div>

                <div className="border-t border-border">
                  <button
                    onClick={() => toggleExpanded(experiment.id)}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 text-sm text-muted-foreground hover:text-foreground hover:bg-secondary/30 transition-colors"
                  >
                    <span>
                      {expandedProjects.has(experiment.id)
                        ? "Show less"
                        : "Show more implementation"}
                    </span>
                    {expandedProjects.has(experiment.id) ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                </div>

                {expandedProjects.has(experiment.id) && (
                  <div className="border-t border-border bg-secondary/20">
                    <div className="p-4">
                      <pre className="text-sm overflow-x-auto">
                        <code className="font-mono text-foreground whitespace-pre-wrap">
                          {experiment.fullCode}
                        </code>
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
