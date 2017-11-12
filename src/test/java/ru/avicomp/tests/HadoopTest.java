package ru.avicomp.tests;

import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;

import cc.fasttext.Args;
import cc.fasttext.FastText;
import cc.fasttext.Main;
import ru.avicomp.TestsBase;
import ru.avicomp.hdfs.HadoopMain;
import ru.avicomp.io.IOStreams;

/**
 * For manual running only
 * Ignore since it requires preconfigured hadoop with test data inside fs.
 * <p>
 * Created by @szuev on 31.10.2017.
 */
@Ignore
public class HadoopTest {

    @Test
    public void hadoopCboxThread4Dim128Ws5Epoch10MinCount5test() throws Exception {
        String hadoopHome = Paths.get(TrainModelTest.class.getResource("/bin").toURI()).getParent().toString();
        Map<String, String> props = new HashMap<>();
        props.put("hadoop.home.dir", hadoopHome);

        IOStreams fs = HadoopMain.createHadoopFS("hdfs://172.16.35.1:54310", "hadoop", Collections.emptyMap(), props);
        TestsBase.LOGGER.info("{}", fs);
        String file = "raw-text-data.txt";
        String dir = "/tmp/out";
        String input = String.format("%s/%s", dir, file);
        String output = String.format("%s/junit.%s.d128.w5.hs", dir, file);

        String[] cmd = TestsBase.cmd("cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5 -verbose 2 -input %s -output %s", input, output);
        Args args = Main.parseArgs(cmd).setIOStreams(fs);
        FastText fasttext = new FastText(args);
        fasttext.train();
        TestsBase.LOGGER.info("Size: {}", fs.size(output + ".bin"));
    }
}
