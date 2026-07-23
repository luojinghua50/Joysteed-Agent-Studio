FROM node:20-alpine AS build

WORKDIR /app
COPY package.json .
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY <<'EOF' /etc/nginx/conf.d/default.conf
server {
    listen 80;

    location / {
        root /usr/share/nginx/html;
        try_files $uri /index.html;
    }

    location /rag/ {
        proxy_pass http://agent-rag:8010/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        client_max_body_size 50m;
    }
}
EOF

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
