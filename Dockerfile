# FROM python:3.12.3-alpine AS compiler
FROM python:3.12.3 AS compiler

WORKDIR /app/

# RUN apk update
# RUN apk add --no-cache gcc g++ gfortran lapack-dev libffi-dev libressl-dev musl-dev libgomp 

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt /app/requirements.txt
RUN pip install -Ur requirements.txt

# FROM python:3.12.3-alpine AS runner
FROM python:3.12.3 AS runner

WORKDIR /app/ 
COPY --from=compiler /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . /app/
EXPOSE 8001
CMD [ "python", "main.py" ]