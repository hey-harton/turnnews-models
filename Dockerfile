FROM public.ecr.aws/lambda/python:3.11

RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .

RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "app.handler" ]