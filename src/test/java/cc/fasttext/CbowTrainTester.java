package cc.fasttext;

import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.fasttext.base.Tests;

/**
 * Not a test.
 * todo: will be removed.
 * Created by @szuev on 25.12.2017.
 */
public class CbowTrainTester {
    public static final Logger LOGGER = LoggerFactory.getLogger(CbowTrainTester.class);

    /**
     * Example of output:
     * <pre>
     * GET_FILE_SIZE                           1              0.0                      0.0
     * READ_DICT                               1              0.16                     0.16
     * IN_MATRIX_CREATE                        1              8.891                    8.891
     * OUT_MATRIX_CREATE                       1              0.0                      0.0
     * FILE_SEEK                               4              0.0                      0.0
     * DIC_GET_LINE                            15084          3.984354282683638E-5     0.601
     * CBOW_CALC                               15084          1.662025987801644E-4     2.507
     * DIC_GET_SUBWORDS_INT                    62144          1.609165808444902E-7     0.01
     * MODEL_UPDATE                            21168          1.1507936507936508E-4    2.436
     * MODEL_COMPUTE_HIDDEN                    18700          2.60427807486631E-5      0.487
     * MODEL_NEGATIVE_SAMPLING                 18700          5.919786096256684E-5     1.107
     * MODEL_GRAD_MUL                          0              NaN                      0.0
     * MODEL_INPUT_ADD_ROW                     18700          4.3475935828877E-5       0.813
     * CREATE_RES_MODEL                        1              0.978                    0.978
     * TRAIN                                   1              14.73                    14.73
     * SAVE_BIN                                1              10.487                   10.487
     * ALL                                     1              25.333                   25.333
     * </pre>
     *
     * @param strs
     * @throws Exception
     */
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
        FastText ft = FastText.DEFAULT_FACTORY.train(args, input);
        ft.saveModel(output);
        Events.ALL.end();
        System.err.println(Events.print());
    }


}
