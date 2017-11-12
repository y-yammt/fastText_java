package com.github.jfasttext;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.junit.After;
import org.junit.Assert;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.fasttext.Args;
import cc.fasttext.FastText;
import cc.fasttext.Main;
import com.google.common.collect.Multimap;
import ru.avicomp.TestsBase;

/**
 * Modified <a href='https://github.com/vinhkhuc/JFastText/blob/master/src/test/java/com/github/jfasttext/JFastTextTest.java'>JFastTextTest</a>
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
        FastText jft = new FastText(Main.createArgs());
        jft.loadModel(TestsBase.DESTINATION_DIR.resolve("supervised.model.bin").toString());

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
        FastText jft = new FastText(Main.createArgs());
        jft.loadModel(TestsBase.DESTINATION_DIR.resolve("supervised.model.bin").toString());
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
        FastText jft = new FastText(Main.createArgs());
        jft.loadModel(TestsBase.DESTINATION_DIR.resolve("supervised.model.bin").toString());

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

    /*

    @Test
    public void test07GetVector() throws Exception {
        JFastText jft = new JFastText();
        jft.loadModel("src/test/resources/models/supervised.model.bin");
        String word = "soccer";
        List<Float> vec = jft.getVector(word);
        System.out.printf("\nWord embedding vector of '%s': %s\n", word, vec);
    }

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
