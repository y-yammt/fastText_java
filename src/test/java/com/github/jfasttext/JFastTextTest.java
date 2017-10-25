package com.github.jfasttext;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.After;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import fasttext.Main;
import ru.avicomp.TestsBase;

/**
 * modified <a href='https://github.com/vinhkhuc/JFastText/blob/master/src/test/java/com/github/jfasttext/JFastTextTest.java'>JFastTextTest</a>
 * Created by @szuev on 24.10.2017.
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class JFastTextTest {

    @After
    public void after() {
        System.out.println("Fin.");
    }

    @Test
    public void test01TrainSupervisedCmd() throws Exception {
        System.out.printf("\nTraining supervised model ...\n");
        Path input = Paths.get(JFastTextTest.class.getResource("/labeled_data.txt").toURI());
        Path out = TestsBase.DESTINATION_DIR.resolve("supervised.model");
        new Main().train(TestsBase.cmd("supervised -input %s -output %s", input, out));
    }

    @Test
    public void test02TrainSkipgramCmd() throws Exception {
        System.out.printf("\nTraining skipgram word-embedding ...\n");
        Path input = Paths.get(JFastTextTest.class.getResource("/unlabeled_data.txt").toURI());
        Path out = TestsBase.DESTINATION_DIR.resolve("skipgram.model");
        new Main().train(TestsBase.cmd("skipgram -input %s -output %s -bucket 100 -minCount 1", input, out));
    }

    @Test
    public void test03TrainCbowCmd() throws Exception {
        System.out.printf("\nTraining cbow word-embedding ...\n");
        Path input = Paths.get(JFastTextTest.class.getResource("/unlabeled_data.txt").toURI());
        Path out = TestsBase.DESTINATION_DIR.resolve("cbow.model");
        new Main().train(TestsBase.cmd("cbow -input %s -output %s -bucket 100 -minCount 1", input, out));
    }

/*
    @Test
    public void test04Predict() throws Exception {
        FastText jft = new FastText();
        jft.loadModel(TestsBase.DESTINATION_DIR.resolve("supervised.model").toString());
        String text = "I like soccer";
        try (InputStream in = new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8.name()))) {
             String predictedLabel = jft.predict(in);
             System.out.printf("\nText: '%s', label: '%s'\n", text, predictedLabel);
             }
    }
*/

    /*
    @Test
    public void test05PredictProba() throws Exception {
        JFastText jft = new JFastText();
        jft.loadModel("src/test/resources/models/supervised.model.bin");
        String text = "What is the most popular sport in the US ?";
        JFastText.ProbLabel predictedProbLabel = jft.predictProba(text);
        System.out.printf("\nText: '%s', label: '%s', probability: %f\n",
                text, predictedProbLabel.label, Math.exp(predictedProbLabel.logProb));
    }

    @Test
    public void test06MultiPredictProba() throws Exception {
        JFastText jft = new JFastText();
        jft.loadModel("src/test/resources/models/supervised.model.bin");
        String text = "Do you like soccer ?";
        System.out.printf("Text: '%s'\n", text);
        for (JFastText.ProbLabel predictedProbLabel: jft.predictProba(text, 2)) {
            System.out.printf("\tlabel: '%s', probability: %f\n",
                    predictedProbLabel.label, Math.exp(predictedProbLabel.logProb));
        }
    }

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
