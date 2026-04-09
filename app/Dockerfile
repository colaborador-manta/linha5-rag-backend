FROM node:20-slim

WORKDIR /app

# Install build tools for better-sqlite3
RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install --production

COPY . .

# Pre-download model during build (cached in image, no download at runtime)
RUN node -e "const{pipeline,env}=require('@xenova/transformers');env.cacheDir='/app/models';pipeline('feature-extraction','Xenova/all-MiniLM-L6-v2',{quantized:true}).then(()=>console.log('Model cached')).catch(e=>console.error(e))"

# Create data directory for SQLite
RUN mkdir -p /app/data

ENV PORT=3001
ENV DB_PATH=/app/data/rag.db

EXPOSE 3001

CMD ["node", "server.js"]
