package ru.avicomp.tests;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.fasttext.Main;
import cc.fasttext.extra.ExtraMain;
import ru.avicomp.TestsBase;
import ru.avicomp.io.IOStreams;

/**
 * For manual running only
 * Ignore since it requires preconfigured hadoop with test data inside fs.
 * May require Xmx jvm-option.
 * Approximately duration for current data and _local_ configuration ~ 15m.
 * <p>
 * Created by @szuev on 31.10.2017.
 */
@Ignore
public class ExtraFSExampleTest {
    private static final Logger LOGGER = LoggerFactory.getLogger(ExtraFSExampleTest.class);

    @Test
    public void hadoopCboxThread4Dim128Ws5Epoch10MinCount5test() throws Exception {
        //System.setProperty("java.library.path", "/opt/hadoop-2.8.1/lib/native");
        IOStreams fs = ExtraMain.getFileSystem();

        String root = "hdfs://hadoop@172.16.35.1:54310";
        String file = "raw-text-data.txt";
        String dir = "/tmp/out";
        String input = String.format("%s%s/%s", root, dir, file);
        String output = String.format("%s%s/junit.%s.d128.w5.hs", root, dir, file);

        long dataSize = fs.size(input);
        LOGGER.info("Size of data(input) file: {}b", dataSize);
        Assert.assertEquals(2272754, dataSize);
        String[] cmd = TestsBase.cmd("cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5 -verbose 2 -input %s -output %s", input, output);
        LOGGER.info("Run");
        Main.setFileSystem(fs);
        Main.run(cmd);
        long binSize = fs.size(output + ".bin");
        long vecSize = fs.size(output + ".vec");
        LOGGER.info("Size of model(bin): {}", binSize);
        LOGGER.info("Size of model(vec): {}", vecSize);
        Assert.assertEquals(1029640131, binSize);
    }
}
