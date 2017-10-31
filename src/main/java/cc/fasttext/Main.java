package cc.fasttext;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;

import ru.avicomp.io.FTReader;

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
     * @param input
     * @throws Exception
     */
    public void predict(String[] input) throws Exception {
        int k = 1;
        if (input.length == 4) {
            k = Integer.parseInt(input[3]);
        } else if (input.length != 3) {
            printPredictUsage();
            System.exit(1);
        }
        boolean printProb = "predict-prob".equalsIgnoreCase(input[0]);
        Args args = createArgs();
        FastText fasttext = new FastText(args);
        fasttext.loadModel(input[1]);
        String infile = input[2];
        PrintStream out = System.out;
        if ("-".equals(infile)) {
            fasttext.predict(System.in, k, printProb);
        } else {
            if (!args.getIOStreams().canRead(infile)) {
                throw new IOException("Input file cannot be opened!");
            }
            try (FTReader in = new FTReader(args.getIOStreams().openInput(infile), args.getCharset())) {
                fasttext._predict(in, out, k, printProb);
            }
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
