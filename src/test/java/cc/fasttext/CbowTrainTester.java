package cc.fasttext;

import cc.fasttext.base.Tests;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Paths;

/**
 * Not a test.
 * todo: will be removed.
 * Use -Devents=true
 *
 * Created by @szuev on 25.12.2017.
 */
public class CbowTrainTester {
    public static final Logger LOGGER = LoggerFactory.getLogger(CbowTrainTester.class);

    public static void main(String... strs) throws Exception {
        Events.ALL.start();
        String input = Paths.get(CbowTrainTester.class.getResource("/text-data.txt").toURI()).toString();
        String output = Tests.DESTINATION_DIR.resolve("res").toString();
        LOGGER.info("Input: {}", input);
        LOGGER.info("Output: {}", output);
        // cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5
        Args args = new Args.Builder()
                .setModel(Args.ModelName.CBOW)
                .setThread(4).setDim(128).setWS(5).setEpoch(10).setMinCount(5).build();
        FastText ft = FastText.train(args, input);
        ft.saveModel(output);
        Events.ALL.end();
        System.err.println(Events.print());
    }


}
