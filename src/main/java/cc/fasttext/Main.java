package cc.fasttext;

import java.io.IOException;
import java.io.InputStream;

/**
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.cc'>main.cc</a>
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.h'>main.h</a>
 */
public class Main {

    private static void printUsage() {
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

    private static void printTestUsage() {
        System.out.print("usage: java -jar fasttext.jar test <model> <test-data> [<k>]\n\n"
                + "  <model>      model filename\n"
                + "  <test-data>  test data filename (if -, read from stdin)\n"
                + "  <k>          (optional; 1 by default) predict top k labels\n");
    }

    private static void printPredictUsage() {
        System.out.print("usage: java -jar fasttext.jar predict[-prob] <model> <test-data> [<k>]\n\n"
                + "  <model>      model filename\n"
                + "  <test-data>  test data filename (if -, read from stdin)\n"
                + "  <k>          (optional; 1 by default) predict top k labels\n");
    }

    private static void printPrintVectorsUsage() {
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
        Args res = createArgs();
        res.parseArgs(args);
        return res;
    }

    /**
     * <pre>{@code void test(const std::vector<std::string>& args) {
     *  if (args.size() < 4 || args.size() > 5) {
     *      printTestUsage();
     *      exit(EXIT_FAILURE);
     *  }
     *  int32_t k = 1;
     *  if (args.size() >= 5) {
     *      k = std::stoi(args[4]);
     *  }
     *  FastText fasttext;
     *  fasttext.loadModel(args[2]);
     *  std::string infile = args[3];
     *  if (infile == "-") {
     *      fasttext.test(std::cin, k);
     *  } else {
     *      std::ifstream ifs(infile);
     *  if (!ifs.is_open()) {
     *      std::cerr << "Test file cannot be opened!" << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     *  fasttext.test(ifs, k);
     *  ifs.close();
     *  }
     *  exit(0);
     * }}</pre>
     *
     * @param input array of args (example: "predict-prob out\dbpedia.bin out\dbpedia.test 7")
     * @throws IOException in case something is wrong while operating with in/out
     */
    public static void test(String[] input) throws IOException {
        int k = 1;
        if (input.length == 4) {
            k = Integer.parseInt(input[3]);
        } else if (input.length != 3) {
            printTestUsage();
            System.exit(1); // todo: should not be exit here
        }

        Args args = createArgs();
        FastText fasttext = new FastText(args);
        fasttext.loadModel(input[1]);
        fasttext.setPrintOut(System.out);
        String infile = input[2];
        if ("-".equals(infile)) {
            fasttext.test(System.in, k);
            return;
        }
        if (!args.getIOStreams().canRead(infile)) {
            throw new IOException("Input file cannot be opened!");
        }
        try (InputStream in = args.getIOStreams().openInput(infile)) {
            fasttext.test(in, k);
        }
    }

    /**
     * <pre>{@code void predict(const std::vector<std::string>& args) {
     *  if (args.size() < 4 || args.size() > 5) {
     *      printPredictUsage();
     *      exit(EXIT_FAILURE);
     *  }
     *  int32_t k = 1;
     *  if (args.size() >= 5) {
     *      k = std::stoi(args[4]);
     *  }
     *  bool print_prob = args[1] == "predict-prob";
     *  FastText fasttext;
     *  fasttext.loadModel(std::string(args[2]));
     *  std::string infile(args[3]);
     *  if (infile == "-") {
     *      fasttext.predict(std::cin, k, print_prob);
     *  } else {
     *      std::ifstream ifs(infile);
     *      if (!ifs.is_open()) {
     *          std::cerr << "Input file cannot be opened!" << std::endl;
     *          exit(EXIT_FAILURE);
     *      }
     *      fasttext.predict(ifs, k, print_prob);
     *      ifs.close();
     *  }
     *  exit(0);
     * }}</pre>
     *
     * @param input array of args (example: "predict-prob out\dbpedia.bin - 7")
     * @throws IOException in case something is wrong with in/out
     */
    public static void predict(String[] input) throws IOException {
        int k = 1;
        if (input.length == 4) {
            k = Integer.parseInt(input[3]);
        } else if (input.length != 3) {
            printPredictUsage();
            System.exit(1); // todo: should not be exit here
        }
        boolean printProb = "predict-prob".equalsIgnoreCase(input[0]);
        Args args = createArgs();
        FastText fasttext = new FastText(args);
        fasttext.loadModel(input[1]);
        fasttext.setPrintOut(System.out);
        String infile = input[2];
        if ("-".equals(infile)) { // read from pipe:
            fasttext.predict(System.in, k, printProb);
            return;
        }
        if (!args.getIOStreams().canRead(infile)) {
            throw new IOException("Input file cannot be opened!");
        }
        try (InputStream in = args.getIOStreams().openInput(infile)) {
            fasttext.predict(in, k, printProb);
        }
    }

    public static void printVectors(String[] args) throws IOException {
        if (args.length != 2) {
            printPrintVectorsUsage();
            System.exit(1);
        }
        FastText fasttext = new FastText(createArgs());
        fasttext.loadModel(args[1]);
        fasttext.printVectors();
    }

    public static void train(String[] args) throws Exception {
        new FastText(parseArgs(args)).trainAndSave();
    }

    public static void main(String... args) {
        try {
            Main.run(args);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void run(String... args) throws Exception {
        if (args.length == 0) {
            printUsage();
            return;
        }
        String command = args[0];
        if ("skipgram".equalsIgnoreCase(command) || "cbow".equalsIgnoreCase(command) || "supervised".equalsIgnoreCase(command)) {
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
