#!/bin/bash

##
## Test to compare training time.
##

if [ $# -eq 0 ]
then
    echo "Usage: ${0} size"
    exit 1
fi

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

hdfs_root="/home/hadoop/hdfs"
hdfs_dir="/tmp/out"
big_file=${hdfs_dir}"/pak-raw-text.txt"
in_file=${hdfs_dir}"/test.${size}${suffix}.txt"
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

echo "Create a input file <${hdfs_root}${in_file}> from the existing big file <${hdfs_root}${big_file}>"
head -c ${bytes} ${hdfs_root}${big_file} > ${hdfs_root}${in_file}
real_size=`ls -la ${hdfs_root}${in_file} | awk -F " " {'print $5'}`
i=0
lim=1000
while (( $real_size != $bytes && $i < $lim ))
do
    sleep 1
    real_size=`ls -la ${hdfs_root}${in_file} | awk -F " " {'print $5'}`
    ((i++))
done
if (( $i == $lim ))
then
    echo "Real size: ${real_size}, but expected: ${bytes}"
    exit $real_size
fi
echo "File created!"

mark_cpp="cpp"
mark_java="jar"
fs_run_cpp="{ time ${fs_cpp} ${fs_args} -input ${hdfs_root}${in_file} -output ${hdfs_root}${model_prefix}${mark_cpp}${model_suffix} ; } >& ${log_prefix}${mark_cpp}${log_suffix}"
fs_run_java="{ time ${fs_java} ${fs_args} -input ${in_file} -output ${model_prefix}${mark_java}${model_suffix} ; } >& ${log_prefix}${mark_java}${log_suffix}"

echo "Run C++: '${fs_run_cpp}'"
eval $fs_run_cpp

echo "Run Java: '${fs_run_java}'"
eval $fs_run_java

exit 0
