 #!/bin/bash
 
/spark/bin/spark-submit \
--master ${SPARK_MASTER_URL} \
 ${SPARK_APPLICATION_PY_LOCATION}