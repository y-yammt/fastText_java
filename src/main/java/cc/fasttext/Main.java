package cc.fasttext;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import ru.avicomp.io.IOStreams;
import ru.avicomp.io.InputStreamSupplier;
import ru.avicomp.io.OutputStreamSupplier;

/**
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.cc'>main.cc</a>
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.h'>main.h</a>
 */
public class Main {

    public static void printUsage() {
        System.out.print("usage: java -jar fasttext.jar <command> <args>\n\n"
                + "The commands supported by fasttext are:\n\n"
                + "  supervised          train a supervised classifier\n"
                + "  test                evaluate a supervised classifier\n"
                + "  predict             predict most likely labels\n"
                + "  predict-prob        predict most likely labels with probabilities\n"
                + "  skipgram            train a skipgram model\n"
                + "  cbow                train a cbow model\n"
                + "  print-vectors       print vectors given a trained model\n");
    }

    public static void printTestUsage() {
        System.out.print("usage: java -jar fasttext.jar test <model> <test-data> [<k>]\n\n"
                + "  <model>      model filename\n"
                + "  <test-data>  test data filename (if -, read from stdin)\n"
                + "  <k>          (optional; 1 by default) predict top k labels\n");
    }

    public static void printPredictUsage() {
        System.out.print("usage: java -jar fasttext.jar predict[-prob] <model> <test-data> [<k>]\n\n"
                + "  <model>      model filename\n"
                + "  <test-data>  test data filename (if -, read from stdin)\n"
                + "  <k>          (optional; 1 by default) predict top k labels\n");
    }

    public static void printPrintVectorsUsage() {
        System.out.print("usage: java -jar fasttext.jar print-vectors <model>\n\n"
                + " <model> model filename\n");
    }

    /**
     * Creates an empty Args
     *
     * @return {@link Args}
     */
    public static Args createArgs() {
        return new Args();
    }

    /**
     * Parses input to Args
     *
     * @param args Array, input
     * @return {@link Args}
     */
    public static Args parseArgs(String... args) {
        Args res = new Args();
        res.parseArgs(args);
        return res;
    }

    public void test(String[] args) throws IOException, Exception {
        int k = 1;
        if (args.length == 3) {
            k = 1;
        } else if (args.length == 4) {
            k = Integer.parseInt(args[3]);
        } else {
            printTestUsage();
            System.exit(1);
        }
        FastText fasttext = new FastText(createArgs());
        fasttext.loadModel(args[1]);
        String infile = args[2];
        if ("-".equals(infile)) {
            fasttext.test(System.in, k);
        } else {
            File file = new File(infile);
            if (!(file.exists() && file.isFile() && file.canRead())) {
                throw new IOException("Test file cannot be opened!");
            }
            fasttext.test(new FileInputStream(file), k);
        }
    }

    public void predict(String[] args) throws Exception {
        int k = 1;
        if (args.length == 3) {
            k = 1;
        } else if (args.length == 4) {
            k = Integer.parseInt(args[3]);
        } else {
            printPredictUsage();
            System.exit(1);
        }
        boolean print_prob = "predict-prob".equalsIgnoreCase(args[0]);
        FastText fasttext = new FastText(createArgs());
        fasttext.loadModel(args[1]);
        IOStreams fs = fasttext.getArgs().getIOStreams();
        fasttext.getArgs().setIOStreams(new IOStreams() {
            @Override
            public InputStreamSupplier createInput(String path) {
                return "-".equals(path) ? () -> System.in : fs.createInput(path);
            }

            @Override
            public OutputStreamSupplier createOutput(String path) {
                return fs.createOutput(path);
            }

            @Override
            public boolean canRead(String path) {
                return "-".equals(path) || fs.canRead(path);
            }

            @Override
            public boolean canWrite(String path) {
                return fs.canRead(path);
            }

            @Override
            public void prepare(String path) throws IOException {
                fs.prepare(path);
            }
        });
        String infile = args[2];
        //if (fasttext.getArgs().getIOStreams().canRead(infile)) {}
        // TODO: implement correct way
        if ("-".equals(infile)) {
            fasttext.predict(System.in, k, print_prob);
        } else {
            File file = new File(infile);
            if (!(file.exists() && file.isFile() && file.canRead())) {
                throw new IOException("Input file cannot be opened!");
            }
            fasttext.predict(new FileInputStream(file), k, print_prob);
        }
    }

    public void printVectors(String[] args) throws IOException {
        if (args.length != 2) {
            printPrintVectorsUsage();
            System.exit(1);
        }
        FastText fasttext = new FastText(createArgs());
        fasttext.loadModel(args[1]);
        fasttext.printVectors();
    }

    public void train(String[] args) throws Exception {
        new FastText(parseArgs(args)).train();
    }

    public static void main(String... args) {
        try {
            new Main().run(args);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void run(String... args) throws Exception {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String command = args[0];
        if ("skipgram".equalsIgnoreCase(command) || "cbow".equalsIgnoreCase(command)
                || "supervised".equalsIgnoreCase(command)) {
            train(args);
        } else if ("test".equalsIgnoreCase(command)) {
            test(args);
        } else if ("print-vectors".equalsIgnoreCase(command)) {
            printVectors(args);
        } else if ("predict".equalsIgnoreCase(command) || "predict-prob".equalsIgnoreCase(command)) {
            predict(args);
        } else {
            printUsage();
        }
    }

}
