package com.github.jfasttext;

import cc.fasttext.Args;
import cc.fasttext.FastText;
import cc.fasttext.Main;
import com.google.common.collect.Multimap;
import com.google.common.primitives.Doubles;
import org.junit.After;
import org.junit.Assert;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.avicomp.TestsBase;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Modified <a href='https://github.com/vinhkhuc/JFastText/blob/master/src/test/java/com/github/jfasttext/JFastTextTest.java'>JFastTextTest</a>
 *
 * Created by @szuev on 24.10.2017.
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class JFastTextTest {
    private static final Logger LOGGER = LoggerFactory.getLogger(JFastTextTest.class);

    @After
    public void after() {
        LOGGER.info("Fin.");
    }

    @Test
    public void test01TrainSupervisedCmd() throws Exception {
        LOGGER.info("Training supervised model ...");
        Path input = Paths.get(JFastTextTest.class.getResource("/labeled_data.txt").toURI());
        Path out = TestsBase.DESTINATION_DIR.resolve("supervised.model");
        Main.train(TestsBase.cmd("supervised -input %s -output %s", input, out));
    }

    @Test
    public void test02TrainSkipgramCmd() throws Exception {
        LOGGER.info("Training skipgram word-embedding ...");
        Path input = Paths.get(JFastTextTest.class.getResource("/unlabeled_data.txt").toURI());
        Path out = TestsBase.DESTINATION_DIR.resolve("skipgram.model");
        Main.train(TestsBase.cmd("skipgram -input %s -output %s -bucket 100 -minCount 1", input, out));
    }

    @Test
    public void test03TrainCbowCmd() throws Exception {
        LOGGER.info("Training cbow word-embedding ...");
        Path input = Paths.get(JFastTextTest.class.getResource("/unlabeled_data.txt").toURI());
        Path out = TestsBase.DESTINATION_DIR.resolve("cbow.model");
        Main.train(TestsBase.cmd("cbow -input %s -output %s -bucket 100 -minCount 1", input, out));
    }

    @Test
    public void test04Predict() throws Exception {
        LOGGER.info("Test predict");
        FastText jft = FastText.load(TestsBase.DESTINATION_DIR.resolve("supervised.model.bin").toString());

        String text = "I like soccer";

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try (InputStream in = new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8.name()));
             PrintStream out = new PrintStream(output, true, StandardCharsets.UTF_8.name())) {
            jft.predict(in, out, 1, false);
        }
        String label = output.toString(StandardCharsets.UTF_8.name()).trim();
        LOGGER.info("Text: '{}', label: '{}'", text, label);
        Assert.assertEquals("Wrong label", Args.DEFAULT_LABEL + "soccer", label);
    }

    @Test
    public void test05PredictProba() throws Exception {
        LOGGER.info("Test predict-proba");
        FastText jft = FastText.load(TestsBase.DESTINATION_DIR.resolve("supervised.model.bin").toString());
        String text = "What is the most popular sport in the US ?";
        // '{__label__football=[-0.6931472]}'
        Multimap<String, Float> predictedProbLabel = jft.predict(text, 1);
        LOGGER.debug("Text: '{}', result: '{}'", text, predictedProbLabel);
        Assert.assertEquals("Wrong result", 1, predictedProbLabel.size());
        String label = predictedProbLabel.entries().stream().map(Map.Entry::getKey).findFirst().orElse(null);
        double probability = predictedProbLabel.entries().stream()
                .mapToDouble(Map.Entry::getValue)
                .map(Math::exp)
                .findFirst().orElse(-1);
        LOGGER.info("Text: '{}', label: '{}', result: {}", text, label, probability);
        Assert.assertEquals("Wrong label", Args.DEFAULT_LABEL + "football", label);
        Assert.assertEquals("Wrong probability", 0.5, probability, 0.0001);
    }

    @Test
    public void test06MultiPredictProba() throws Exception {
        LOGGER.info("Test multi-predict-proba");
        FastText jft = FastText.load(TestsBase.DESTINATION_DIR.resolve("supervised.model.bin").toString());

        String text = "Do you like soccer ?";
        Multimap<String, Float> predictedProbLabel = jft.predict(text, 2);
        LOGGER.debug("Text: '{}', result: '{}'", text, predictedProbLabel);
        Assert.assertEquals("Wrong result", 2, predictedProbLabel.size());
        Map<String, Double> res = predictedProbLabel.entries().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> Math.exp(e.getValue())));
        LOGGER.debug("Map result: '{}'", res);
        // __label__soccer 0.5 __label__football 0.498047
        Map<String, Double> expected = new HashMap<>();
        expected.put("soccer", 0.5);
        expected.put("football", 0.498047);
        expected.forEach((k, v) -> {
            Assert.assertTrue("Can't find label '" + k + "'", res.containsKey(Args.DEFAULT_LABEL + k));
            Assert.assertEquals("Wrong probability for label '" + k + "'", v, res.get(Args.DEFAULT_LABEL + k), 0.0001);
        });
    }

    @Test
    public void test07GetVector() throws Exception {
        List<Double> expected = Doubles.asList(-0.007984138, -0.0016172099, 0.007518901, 0.005489656, 3.875666E-5, -0.0034641975, -0.0098630795, -0.0052505974,
                -0.0027585188, -0.0072525786, -0.003865521, 0.0021957296, -0.0051714815, -5.4303935E-4, -0.008342951, 7.993335E-4, 0.0061263866, 0.006974083,
                -0.00813139, 0.0012676214, 0.008962845, -0.004499837, -0.003216704, -4.0615757E-4, 0.007194679, 3.8332146E-4, 0.004404881, -0.009127971,
                -0.004435753, 0.0055470644, -0.0061344043, 0.0018859841, -0.006747924, 0.002202363, -2.423497E-4, 0.008315044, -5.801832E-4, 0.00535778,
                0.0064756647, 3.9104614E-4, -0.007072684, -0.0011439206, -0.009660971, 0.009170009, -0.008711385, -0.0076893945, 0.00798731, 0.002366419,
                -1.4621246E-4, 0.007029484, -0.0060989284, -5.319206E-4, 0.0014283261, -0.0042950558, -0.008760027, -0.008183239, -0.0014670147, -0.005777317,
                0.0068037617, -0.010046145, 0.004997755, 0.0036687413, 0.009267032, -0.009153373, -0.0064546084, -0.007426012, 0.0044944924, -0.006416458,
                -0.003338646, 0.0064553134, -0.0060658683, -0.0047504655, -0.008723757, -0.003118516, 0.0051919487, -0.005962505, -9.916733E-4, -0.0014734096,
                4.0317033E-4, 0.004028524, -0.00955603, 0.002795295, -0.0013588008, -0.0022180455, -0.009875522, 0.008161922, -0.00844231, 0.007907908,
                0.0013905984, 0.0063601835, 0.006373907, 0.0057821167, 0.0056400453, 0.008395423, 0.002067171, 0.005855603, -7.2561664E-4, -0.008864928,
                0.0049015884, 0.009060863);

        LOGGER.info("Test get vector");
        FastText jft = FastText.load(TestsBase.DESTINATION_DIR.resolve("supervised.model.bin").toString());
        String word = "soccer";
        List<Double> vec = jft.getWordVector(word).getData()
                .stream().mapToDouble(d -> d).boxed().collect(Collectors.toList());
        LOGGER.debug("Word embedding vector of '{}': {}", word, vec);
        TestsBase.compareVectors(expected, vec, 0.1);
    }

    /*
    @Test
    public void test08ModelInfo() throws Exception {
        System.out.printf("\nSupervised model information:\n");
        JFastText jft = new JFastText();
        jft.loadModel("src/test/resources/models/supervised.model.bin");
        System.out.printf("\tnumber of words = %d\n", jft.getNWords());
        System.out.printf("\twords = %s\n", jft.getWords());
        System.out.printf("\tlearning rate = %g\n", jft.getLr());
        System.out.printf("\tdimension = %d\n", jft.getDim());
        System.out.printf("\tcontext window size = %d\n", jft.getContextWindowSize());
        System.out.printf("\tepoch = %d\n", jft.getEpoch());
        System.out.printf("\tnumber of sampled negatives = %d\n", jft.getNSampledNegatives());
        System.out.printf("\tword ngrams = %d\n", jft.getWordNgrams());
        System.out.printf("\tloss name = %s\n", jft.getLossName());
        System.out.printf("\tmodel name = %s\n", jft.getModelName());
        System.out.printf("\tnumber of buckets = %d\n", jft.getBucket());
        System.out.printf("\tlabel prefix = %s\n\n", jft.getLabelPrefix());
    }

    @Test
    public void test09ModelUnloading() throws Exception {
        JFastText jft = new JFastText();
        System.out.println("\nLoading model ...");
        jft.loadModel("src/test/resources/models/supervised.model.bin");
        System.out.println("Unloading model ...");
        jft.unloadModel();
    }

    */
}
