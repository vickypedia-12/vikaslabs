"use client"

import { useState } from 'react'
import { ChevronDown, ChevronUp, ExternalLink, Github } from 'lucide-react'

const experiments = [
  {
    id: 'llm-finetuning',
    name: 'LLM Fine-tuning Pipeline',
    description: 'End-to-end pipeline for fine-tuning language models with custom datasets using LoRA and QLoRA techniques.',
    preview: `import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def setup_lora_model(model_name, r=16, alpha=32):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    return get_peft_model(model, lora_config)`,
    fullCode: `import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd

class LLMFineTuner:
    def __init__(self, model_name, output_dir="./results"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
    def setup_lora_model(self, r=16, alpha=32):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(model, lora_config)
        return self.model
    
    def prepare_dataset(self, data_path):
        df = pd.read_csv(data_path)
        
        def format_prompt(row):
            return f"### Instruction:\\n{row['instruction']}\\n\\n### Response:\\n{row['response']}"
        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda x: {"text": format_prompt(x)})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def train(self, dataset, epochs=3, batch_size=4):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            warmup_steps=100,
            report_to="tensorboard"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        
    def inference(self, prompt, max_length=256):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)`,
    github: 'https://github.com/username/llm-finetuning',
    demo: 'https://demo.example.com/llm'
  },
  {
    id: 'api-rate-limiter',
    name: 'API Rate Limiter',
    description: 'Distributed rate limiting system with Redis backend supporting sliding window and token bucket algorithms.',
    preview: `import redis
import time
from typing import Optional

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        now = time.time()
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now): now})
        
        # Set expiration
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        return results[1] < limit`,
    fullCode: `import redis
import time
import json
from typing import Optional, Dict, Any
from enum import Enum

class RateLimitAlgorithm(Enum):
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"

class RateLimiter:
    def __init__(self, redis_client: redis.Redis, algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW):
        self.redis = redis_client
        self.algorithm = algorithm
    
    def is_allowed(self, key: str, limit: int, window: int, tokens: int = 1) -> Dict[str, Any]:
        if self.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return self._sliding_window_check(key, limit, window, tokens)
        elif self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._token_bucket_check(key, limit, window, tokens)
        else:
            return self._fixed_window_check(key, limit, window, tokens)
    
    def _sliding_window_check(self, key: str, limit: int, window: int, tokens: int) -> Dict[str, Any]:
        now = time.time()
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Execute pipeline
        results = pipeline.execute()
        current_count = results[1]
        
        if current_count + tokens <= limit:
            # Add current request(s)
            pipeline = self.redis.pipeline()
            for i in range(tokens):
                pipeline.zadd(key, {f"{now}_{i}": now})
            pipeline.expire(key, window)
            pipeline.execute()
            
            return {
                "allowed": True,
                "remaining": limit - current_count - tokens,
                "reset_time": now + window,
                "retry_after": None
            }
        else:
            # Get the oldest request time to calculate retry_after
            oldest_requests = self.redis.zrange(key, 0, 0, withscores=True)
            retry_after = window if not oldest_requests else oldest_requests[0][1] + window - now
            
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": now + window,
                "retry_after": max(0, retry_after)
            }
    
    def _token_bucket_check(self, key: str, capacity: int, refill_rate: int, tokens: int) -> Dict[str, Any]:
        now = time.time()
        bucket_key = f"bucket:{key}"
        
        # Get current bucket state
        bucket_data = self.redis.get(bucket_key)
        
        if bucket_data:
            bucket = json.loads(bucket_data)
            last_refill = bucket["last_refill"]
            current_tokens = bucket["tokens"]
        else:
            last_refill = now
            current_tokens = capacity
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - last_refill
        tokens_to_add = int(time_elapsed * refill_rate)
        current_tokens = min(capacity, current_tokens + tokens_to_add)
        
        if current_tokens >= tokens:
            # Consume tokens
            current_tokens -= tokens
            
            # Update bucket
            bucket_data = {
                "tokens": current_tokens,
                "last_refill": now,
                "capacity": capacity,
                "refill_rate": refill_rate
            }
            self.redis.set(bucket_key, json.dumps(bucket_data), ex=3600)
            
            return {
                "allowed": True,
                "remaining": current_tokens,
                "reset_time": now + (capacity - current_tokens) / refill_rate,
                "retry_after": None
            }
        else:
            # Calculate retry after
            tokens_needed = tokens - current_tokens
            retry_after = tokens_needed / refill_rate
            
            return {
                "allowed": False,
                "remaining": current_tokens,
                "reset_time": now + (capacity - current_tokens) / refill_rate,
                "retry_after": retry_after
            }
    
    def _fixed_window_check(self, key: str, limit: int, window: int, tokens: int) -> Dict[str, Any]:
        now = time.time()
        window_start = int(now // window) * window
        window_key = f"{key}:{window_start}"
        
        current_count = self.redis.get(window_key)
        current_count = int(current_count) if current_count else 0
        
        if current_count + tokens <= limit:
            # Increment counter
            pipeline = self.redis.pipeline()
            pipeline.incrby(window_key, tokens)
            pipeline.expire(window_key, window)
            pipeline.execute()
            
            return {
                "allowed": True,
                "remaining": limit - current_count - tokens,
                "reset_time": window_start + window,
                "retry_after": None
            }
        else:
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": window_start + window,
                "retry_after": window_start + window - now
            }
    
    def get_stats(self, key: str) -> Dict[str, Any]:
        if self.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return {"current_count": self.redis.zcard(key)}
        elif self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            bucket_data = self.redis.get(f"bucket:{key}")
            if bucket_data:
                return json.loads(bucket_data)
            return {"tokens": 0, "last_refill": 0}
        else:
            now = time.time()
            window_start = int(now // 60) * 60  # Assuming 60s window
            window_key = f"{key}:{window_start}"
            return {"current_count": int(self.redis.get(window_key) or 0)}`,
    github: 'https://github.com/username/api-rate-limiter',
    demo: 'https://demo.example.com/rate-limiter'
  },
  {
    id: 'data-pipeline',
    name: 'Automated Data Pipeline',
    description: 'Scalable ETL pipeline with Apache Airflow for processing large datasets with error handling and monitoring.',
    preview: `from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

def extract_data(**context):
    # Extract data from multiple sources
    df = pd.read_sql_query(
        "SELECT * FROM raw_data WHERE date >= %s",
        connection,
        params=[context['ds']]
    )
    return df.to_json()

def transform_data(**context):
    df = pd.read_json(context['ti'].xcom_pull(task_ids='extract'))
    
    # Data cleaning and transformation
    df['processed_date'] = pd.to_datetime(df['date'])
    df = df.dropna().drop_duplicates()
    
    return df.to_json()`,
    fullCode: `from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import pandas as pd
import boto3
import logging
import json

# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Create DAG
dag = DAG(
    'data_pipeline_etl',
    default_args=default_args,
    description='Automated ETL pipeline for data processing',
    schedule_interval=timedelta(hours=1),
    max_active_runs=1,
    tags=['etl', 'data-pipeline']
)

class DataPipelineProcessor:
    def __init__(self, aws_access_key, aws_secret_key, region='us-east-1'):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_from_s3(self, bucket_name, prefix, **context):
        """Extract data from S3 bucket"""
        try:
            objects = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            dataframes = []
            for obj in objects.get('Contents', []):
                if obj['Key'].endswith('.csv'):
                    response = self.s3_client.get_object(
                        Bucket=bucket_name,
                        Key=obj['Key']
                    )
                    df = pd.read_csv(response['Body'])
                    df['source_file'] = obj['Key']
                    dataframes.append(df)
            
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                self.logger.info(f"Extracted {len(combined_df)} rows from S3")
                return combined_df.to_json()
            else:
                raise ValueError("No CSV files found in S3 bucket")
                
        except Exception as e:
            self.logger.error(f"Error extracting from S3: {str(e)}")
            raise
    
    def extract_from_database(self, query, **context):
        """Extract data from database"""
        try:
            import psycopg2
            
            connection = psycopg2.connect(
                host=context['params']['db_host'],
                database=context['params']['db_name'],
                user=context['params']['db_user'],
                password=context['params']['db_password']
            )
            
            df = pd.read_sql_query(query, connection)
            connection.close()
            
            self.logger.info(f"Extracted {len(df)} rows from database")
            return df.to_json()
            
        except Exception as e:
            self.logger.error(f"Error extracting from database: {str(e)}")
            raise
    
    def transform_data(self, **context):
        """Transform and clean data"""
        try:
            # Get data from previous task
            raw_data = context['ti'].xcom_pull(task_ids='extract_s3_data')
            df = pd.read_json(raw_data)
            
            # Data cleaning
            df = df.dropna()
            df = df.drop_duplicates()
            
            # Data transformation
            df['processed_timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['processed_timestamp'].dt.date
            df['hour'] = df['processed_timestamp'].dt.hour
            
            # Feature engineering
            df['value_normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
            df['category_encoded'] = pd.Categorical(df['category']).codes
            
            # Data validation
            if df.empty:
                raise ValueError("DataFrame is empty after transformation")
            
            if df['value'].isnull().all():
                raise ValueError("All values are null after transformation")
            
            # Create data quality metrics
            quality_metrics = {
                'total_rows': len(df),
                'null_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                'duplicate_percentage': df.duplicated().sum() / len(df) * 100,
                'processing_date': datetime.now().isoformat()
            }
            
            # Store quality metrics
            context['ti'].xcom_push(key='quality_metrics', value=quality_metrics)
            
            self.logger.info(f"Transformed {len(df)} rows")
            return df.to_json()
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def load_to_warehouse(self, table_name, **context):
        """Load data to data warehouse"""
        try:
            import psycopg2
            from sqlalchemy import create_engine
            
            # Get transformed data
            transformed_data = context['ti'].xcom_pull(task_ids='transform_data')
            df = pd.read_json(transformed_data)
            
            # Create database connection
            engine = create_engine(
                f"postgresql://{context['params']['warehouse_user']}:"
                f"{context['params']['warehouse_password']}@"
                f"{context['params']['warehouse_host']}/"
                f"{context['params']['warehouse_db']}"
            )
            
            # Load data
            df.to_sql(
                table_name,
                engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            self.logger.info(f"Loaded {len(df)} rows to {table_name}")
            return {"rows_loaded": len(df), "table": table_name}
            
        except Exception as e:
            self.logger.error(f"Error loading to warehouse: {str(e)}")
            raise
    
    def send_notification(self, **context):
        """Send completion notification"""
        try:
            quality_metrics = context['ti'].xcom_pull(
                task_ids='transform_data',
                key='quality_metrics'
            )
            
            load_result = context['ti'].xcom_pull(task_ids='load_to_warehouse')
            
            message = f"""
            🎉 Data Pipeline Completed Successfully!
            
            📊 Quality Metrics:
            • Total rows processed: {quality_metrics['total_rows']}
            • Null percentage: {quality_metrics['null_percentage']:.2f}%
            • Duplicate percentage: {quality_metrics['duplicate_percentage']:.2f}%
            
            📈 Load Results:
            • Rows loaded: {load_result['rows_loaded']}
            • Target table: {load_result['table']}
            
            ⏰ Processing date: {quality_metrics['processing_date']}
            """
            
            # Send Slack notification (you would configure this)
            self.logger.info("Pipeline completed successfully")
            return message
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            raise

# Initialize processor
processor = DataPipelineProcessor(
    aws_access_key='your_access_key',
    aws_secret_key='your_secret_key'
)

# Define tasks
extract_s3_task = PythonOperator(
    task_id='extract_s3_data',
    python_callable=processor.extract_from_s3,
    op_kwargs={
        'bucket_name': 'your-data-bucket',
        'prefix': 'raw-data/'
    },
    dag=dag
)

extract_db_task = PythonOperator(
    task_id='extract_db_data',
    python_callable=processor.extract_from_database,
    op_kwargs={
        'query': 'SELECT * FROM raw_table WHERE created_at >= {{ ds }}'
    },
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=processor.transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=processor.load_to_warehouse,
    op_kwargs={'table_name': 'processed_data'},
    dag=dag
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=processor.send_notification,
    dag=dag
)

# Set task dependencies
[extract_s3_task, extract_db_task] >> transform_task >> load_task >> notify_task`,
    github: 'https://github.com/username/data-pipeline',
    demo: 'https://demo.example.com/pipeline'
  },
  {
    id: 'ml-deployment',
    name: 'ML Model Deployment',
    description: 'Production-ready ML model serving with FastAPI, Docker, and monitoring for real-time predictions.',
    preview: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="ML Model API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]
    
class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_version: str

# Load model
model = joblib.load("model.pkl")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version="v1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))`,
    fullCode: `from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import time
import asyncio
from datetime import datetime
import redis
import json
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total ML requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ml_request_duration_seconds', 'ML request duration')
PREDICTION_ACCURACY = Histogram('ml_prediction_accuracy', 'Model prediction accuracy')

# Security
security = HTTPBearer()

app = FastAPI(
    title="ML Model API",
    version="1.0.0",
    description="Production-ready ML model serving API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Input features for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")

class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_version: str
    prediction_id: str
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    batch_features: List[List[float]]
    model_version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_id: str
    processing_time: float

class ModelHealth(BaseModel):
    status: str
    model_version: str
    last_trained: datetime
    accuracy: float
    uptime: float

class MLModelManager:
    def __init__(self):
        self.models = {}
        self.current_version = "v1.0.0"
        self.load_models()
        self.start_time = time.time()
        
    def load_models(self):
        """Load all available model versions"""
        try:
            model_dir = "models"
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pkl'):
                        version = file.replace('.pkl', '')
                        self.models[version] = joblib.load(os.path.join(model_dir, file))
                        logger.info(f"Loaded model version: {version}")
            else:
                # Load default model
                self.models[self.current_version] = joblib.load("model.pkl")
                logger.info(f"Loaded default model: {self.current_version}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_model(self, version: str = None):
        """Get model by version"""
        version = version or self.current_version
        if version not in self.models:
            raise ValueError(f"Model version {version} not found")
        return self.models[version]
    
    def predict(self, features: np.ndarray, version: str = None) -> Dict[str, Any]:
        """Make prediction with specified model version"""
        model = self.get_model(version)
        
        try:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            probability = probabilities.max()
            
            return {
                "prediction": float(prediction),
                "probability": float(probability),
                "model_version": version or self.current_version
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def batch_predict(self, batch_features: List[List[float]], version: str = None) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        model = self.get_model(version)
        
        try:
            features_array = np.array(batch_features)
            predictions = model.predict(features_array)
            probabilities = model.predict_proba(features_array)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "prediction": float(pred),
                    "probability": float(prob.max()),
                    "model_version": version or self.current_version
                })
            
            return results
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise

# Initialize model manager
model_manager = MLModelManager()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    # In production, implement proper token verification
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "uptime": time.time() - model_manager.start_time
    }

@app.get("/model/health", response_model=ModelHealth)
async def model_health():
    """Get model health status"""
    return ModelHealth(
        status="healthy",
        model_version=model_manager.current_version,
        last_trained=datetime.now(),  # This would be loaded from model metadata
        accuracy=0.95,  # This would be calculated from validation data
        uptime=time.time() - model_manager.start_time
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Make a single prediction"""
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.features) == 0:
            raise HTTPException(status_code=400, detail="Features cannot be empty")
        
        # Check cache first
        cache_key = f"prediction:{hash(str(request.features))}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            logger.info("Returning cached prediction")
            result = json.loads(cached_result)
            REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="cached").inc()
            return PredictionResponse(**result)
        
        # Make prediction
        features = np.array(request.features).reshape(1, -1)
        result = model_manager.predict(features, request.model_version)
        
        # Generate prediction ID
        prediction_id = f"pred_{int(time.time() * 1000)}"
        
        # Create response
        response = PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            model_version=result["model_version"],
            prediction_id=prediction_id,
            timestamp=datetime.now()
        )
        
        # Cache result
        redis_client.setex(
            cache_key,
            300,  # 5 minutes TTL
            json.dumps(response.dict(), default=str)
        )
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            prediction_id,
            request.features,
            result["prediction"],
            result["probability"]
        )
        
        # Update metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
        REQUEST_DURATION.observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Make batch predictions"""
    start_time = time.time()
    
    try:
        if len(request.batch_features) == 0:
            raise HTTPException(status_code=400, detail="Batch features cannot be empty")
        
        # Make batch predictions
        results = model_manager.batch_predict(
            request.batch_features,
            request.model_version
        )
        
        # Create response objects
        predictions = []
        batch_id = f"batch_{int(time.time() * 1000)}"
        
        for i, result in enumerate(results):
            prediction_id = f"{batch_id}_{i}"
            predictions.append(PredictionResponse(
                prediction=result["prediction"],
                probability=result["probability"],
                model_version=result["model_version"],
                prediction_id=prediction_id,
                timestamp=datetime.now()
            ))
        
        processing_time = time.time() - start_time
        
        # Log batch prediction
        background_tasks.add_task(
            log_batch_prediction,
            batch_id,
            len(request.batch_features),
            processing_time
        )
        
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="success").inc()
        REQUEST_DURATION.observe(processing_time)
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="error").inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/models")
async def list_models():
    """List available model versions"""
    return {
        "available_versions": list(model_manager.models.keys()),
        "current_version": model_manager.current_version
    }

@app.post("/models/{version}/activate")
async def activate_model(version: str, token: str = Depends(verify_token)):
    """Activate a specific model version"""
    if version not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    old_version = model_manager.current_version
    model_manager.current_version = version
    
    logger.info(f"Activated model version: {version} (was: {old_version})")
    
    return {
        "message": f"Activated model version: {version}",
        "previous_version": old_version,
        "current_version": version
    }

async def log_prediction(prediction_id: str, features: List[float], prediction: float, probability: float):
    """Log prediction for monitoring and analysis"""
    log_data = {
        "prediction_id": prediction_id,
        "features": features,
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.now().isoformat()
    }
    
    # In production, this would go to a proper logging system
    logger.info(f"Prediction logged: {log_data}")

async def log_batch_prediction(batch_id: str, batch_size: int, processing_time: float):
    """Log batch prediction for monitoring"""
    log_data = {
        "batch_id": batch_id,
        "batch_size": batch_size,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Batch prediction logged: {log_data}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`,
    github: 'https://github.com/username/ml-deployment',
    demo: 'https://demo.example.com/ml-api'
  }
]

export default function ExperimentsSection() {
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set())

  const toggleExpanded = (projectId: string) => {
    setExpandedProjects(prev => {
      const newSet = new Set(prev)
      if (newSet.has(projectId)) {
        newSet.delete(projectId)
      } else {
        newSet.add(projectId)
      }
      return newSet
    })
  }

  return (
    <section id="experiments" className="py-20 bg-background">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-foreground mb-4">
            My Experiments
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Real-world AI/ML and API projects showcasing production-ready implementations
            with comprehensive code examples and deployment strategies.
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
                    {experiment.name.toLowerCase().replace(/\s+/g, '-')}.py
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
                      {expandedProjects.has(experiment.id) ? 'Show less' : 'Show more implementation'}
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
  )
}