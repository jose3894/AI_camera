FROM spark-base:latest

COPY spark-submit.sh /

ENV SPARK_MASTER_URL="spark://spark-master:7077"
ENV SPARK_SUBMIT_ARGS=""
ENV SPARK_APPLICATION_ARGS ""
ENV SPARK_APPLICATION_PY_LOCATION /opt/spark-apps/mnist_data_setup.py


CMD ["/bin/bash", "/spark-submit.sh"]
