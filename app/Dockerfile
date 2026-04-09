FROM node:20-slim

WORKDIR /app

RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install --production

COPY . .

# Pre-download model during build (ESM script)
RUN node download-model.mjs

RUN mkdir -p /app/data

ENV PORT=3001
ENV DB_PATH=/app/data/rag.db

EXPOSE 3001

CMD ["node", "server.js"]
