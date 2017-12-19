#!/bin/bash

##
## Comparing training time (Java vs C++).
## To fix script use "sed -i -e 's/\r$//' /file"
## Run example: nohup ./fastText/compare_test.sh 1g -jar &> /home/avicomp/fastText/logs/nohup.log &
##

if [ $# -eq 0 ]
then
    printf "Usage: ${0} size [-jar|cpp]\nExample: ${0} 1g -cpp\n"
    exit 1
fi

now=`date '+%Y-%m-%d-%H:%M:%S'`
echo "Now: $now"
arg="$1"
max_gb=10
max_bytes=$((${max_gb} * 1024 * 1024 * 1024))
suffix=""
size=""
bytes=0
if [[ ${arg} =~ ^[0-9]+((g|G)(b|B)*)*$ ]]
then
    size=${arg//[^0-9]/}
    bytes=$((1024 * 1024 * 1024 * ${size}))
    suffix="g"
elif [[ ${arg} =~ ^[0-9]+((m|M)(b|B)*)*$ ]]
then
    size=${arg//[^0-9]/}
    bytes=$((1024 * 1024 * ${size}))
    suffix="m"
else
    echo "Wrong input: '${arg}' not a number"
    exit 1
fi
if (( $bytes > $max_bytes ))
then
    echo "Too big file size specified"
    exit -1
fi
echo "Input file size ${bytes}b (${size}${suffix})"

is_run_cpp=false
is_run_java=false
if [ "$#" -eq 1 ]
then
    is_run_cpp=true
    is_run_java=true
elif [[ $2 =~ ^(-)*(j|J)(a|A)(r|R)$ ]]
then
    is_run_java=true
elif [[ $2 =~ ^(-)*(c|C)(p|P)(p|P)$ ]]
then
    is_run_cpp=true
else
    echo "Illegal argument: '${2}'"
    exit 2
fi

PREPARE=true
HADOOP_ROOT="/home/hadoop/hdfs"
HADOOP_DIR="/tmp/out"
BIG_FILE=${HADOOP_DIR}"/pak-raw-text.txt"
BIG_FILE_REF=${HADOOP_ROOT}${BIG_FILE}
IN_FILE=${HADOOP_DIR}"/test.${size}${suffix}.txt"
IN_FILE_REF=${HADOOP_ROOT}${IN_FILE}

model_prefix=${HADOOP_DIR}"/test${size}${suffix}."
model_suffix=".cbox.d128.w5.hs"
LOG_DIR="/home/avicomp/fastText/logs"
log_prefix=${LOG_DIR}"/fasttext.${size}${suffix}."
log_suffix=".log"

JAVA_XMX_GB=10
HADOOP_VM_OPTS="java.library.path=/opt/hadoop-2.8.1/lib/native"
HADOOP_HOST="172.16.35.1"
HADOOP_PORT="54310"
HADOOP_USER="hadoop"
hadoop_fs_prefix="hdfs://${HADOOP_USER}@${HADOOP_HOST}:${HADOOP_PORT}"

FS_ARGS="cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5"
FS_CPP_HOME="/home/avicomp/fastText/build/"
fs_cpp_exe=${FS_CPP_HOME}"fasttext"
fs_jar_file="fasttext-hadoop-jar-with-dependencies.jar"
FS_JAVA_HOME="/home/avicomp/fastText/java/"
fs_java_exe="java -jar -Xmx${JAVA_XMX_GB}G -D${HADOOP_VM_OPTS} ${FS_JAVA_HOME}${fs_jar_file}"

MARK_CPP="cpp"
MARK_JAVA="jar"
logs_cpp=${log_prefix}${MARK_CPP}${log_suffix}
logs_java=${log_prefix}${MARK_JAVA}${log_suffix}

if $PREPARE
then
    if [ ! -f $BIG_FILE_REF ]
    then
        echo "No big file <${BIG_FILE_REF}> in system!"
        exit 3
    fi

    if [[ -f $IN_FILE_REF && `ls -la ${IN_FILE_REF} | awk -F " " {'print $5'}` -eq $bytes ]]
    then
        echo "File <${IN_FILE_REF}> already exists"
    else
        echo "Create a input file <${IN_FILE_REF}> from the existing big file <${BIG_FILE_REF}>"
        head -c ${bytes} ${BIG_FILE_REF} > ${IN_FILE_REF}
        real_size=`ls -la ${IN_FILE_REF} | awk -F " " {'print $5'}`
        i=0
        lim=1000
        while (( $real_size != $bytes && $i < $lim ))
        do
            sleep 1
            real_size=`ls -la ${IN_FILE_REF} | awk -F " " {'print $5'}`
            ((i++))
        done
        if (( $i == $lim ))
        then
            echo "Real size: ${real_size}, but expected: ${bytes}"
            exit $real_size
        fi
        echo "File created!"
    fi
fi

fs_run_cpp="{ time ${fs_cpp_exe} ${FS_ARGS} -input ${IN_FILE_REF} -output ${HADOOP_ROOT}${model_prefix}${MARK_CPP}${model_suffix} ; } >& ${logs_cpp}"
fs_run_java="{ time ${fs_java_exe} ${FS_ARGS} -input ${hadoop_fs_prefix}${IN_FILE} -output ${hadoop_fs_prefix}${model_prefix}${MARK_JAVA}${model_suffix} ; } >& ${logs_java}"

if $is_run_cpp
then
    echo "Run C++: '${fs_run_cpp}'"
    eval $fs_run_cpp
    echo "date=${now}" >> $logs_cpp
fi

if $is_run_java
then
    echo "Run Java: '${fs_run_java}'"
    eval $fs_run_java
    echo "date=${now}" >> $logs_java
fi

exit 0
