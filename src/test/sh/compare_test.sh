#!/bin/bash

##
## Comparing training time (Java vs C++).
## To fix script use "sed -i -e 's/\r$//' /file"
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


hdfs_root="/home/hadoop/hdfs"
hdfs_dir="/tmp/out"
big_file=${hdfs_dir}"/pak-raw-text.txt"
big_file_ref=${hdfs_root}${big_file}
in_file=${hdfs_dir}"/test.${size}${suffix}.txt"
in_file_ref=${hdfs_root}${in_file}

if [ ! -f $big_file_ref ]
then
    echo "No big file <${big_file_ref}> in system!"
    exit 3
fi

model_prefix=${hdfs_dir}"/test${size}${suffix}."
model_suffix=".cbox.d128.w5.hs"
log_dir="/tmp/fasttext-test"
log_prefix=${log_dir}"/fasttext.${size}${suffix}."
log_suffix=".log"

java_XMX_GB=5
java_vm_opts="java.library.path=/opt/hadoop-2.8.1/lib/native"
hadoop_url="hdfs://172.16.35.1:54310"
hadoop_user="hadoop"

fs_args="cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5"
fs_cpp="/tmp/fasttext-test/fastText/fasttext"
fs_jar_file="/tmp/fasttext-test/fasttext-hadoop-jar-with-dependencies.jar"
fs_java="java -jar -Xmx${java_XMX_GB}G -D${java_vm_opts} ${fs_jar_file} -hadoop-url ${hadoop_url} -hadoop-user ${hadoop_user}"

if [[ -f $in_file_ref && `ls -la ${in_file_ref} | awk -F " " {'print $5'}` -eq $bytes ]]
then
    echo "File <${in_file_ref}> already exists"
else
    echo "Create a input file <${in_file_ref}> from the existing big file <${big_file_ref}>"
    head -c ${bytes} ${big_file_ref} > ${in_file_ref}
    real_size=`ls -la ${in_file_ref} | awk -F " " {'print $5'}`
    i=0
    lim=1000
    while (( $real_size != $bytes && $i < $lim ))
    do
        sleep 1
        real_size=`ls -la ${in_file_ref} | awk -F " " {'print $5'}`
        ((i++))
    done
    if (( $i == $lim ))
    then
        echo "Real size: ${real_size}, but expected: ${bytes}"
        exit $real_size
    fi
    echo "File created!"
fi

mark_cpp="cpp"
mark_java="jar"
logs_cpp=${log_prefix}${mark_cpp}${log_suffix}
logs_java=${log_prefix}${mark_java}${log_suffix}
fs_run_cpp="{ time ${fs_cpp} ${fs_args} -input ${in_file_ref} -output ${hdfs_root}${model_prefix}${mark_cpp}${model_suffix} ; } >& ${logs_cpp}"
fs_run_java="{ time ${fs_java} ${fs_args} -input ${in_file} -output ${model_prefix}${mark_java}${model_suffix} ; } >& ${logs_java}"

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
