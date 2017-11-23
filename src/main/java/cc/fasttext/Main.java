package cc.fasttext;

import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;

/**
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.cc'>main.cc</a>
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.h'>main.h</a>
 */
public class Main {

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
     * @throws IOException              in case something is wrong while operating with in/out
     * @throws IllegalArgumentException in case wrong input
     */
    public static void test(String[] input) throws IOException, IllegalArgumentException {
        int k = 1;
        if (input.length == 4) {
            k = Integer.parseInt(input[3]);
        } else if (input.length != 3) {
            throw Usage.TEST.toException();
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
     * @throws IOException              in case something is wrong with in/out
     * @throws IllegalArgumentException wrong inputs
     */
    public static void predict(String[] input) throws IOException, IllegalArgumentException {
        int k = 1;
        if (input.length == 4) {
            k = Integer.parseInt(input[3]);
        } else if (input.length != 3) {
            throw Usage.PREDICT.toException();
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

    /**
     * <pre>{@code void printWordVectors(const std::vector<std::string> args) {
     *  if (args.size() != 3) {
     *      printPrintWordVectorsUsage();
     *      exit(EXIT_FAILURE);
     *  }
     *  FastText fasttext;
     *  fasttext.loadModel(std::string(args[2]));
     *  std::string word;
     *  Vector vec(fasttext.getDimension());
     *  while (std::cin >> word) {
     *      fasttext.getWordVector(vec, word);
     *      std::cout << word << " " << vec << std::endl;
     *  }
     *  exit(0);
     * }}</pre>
     *
     * @param input
     * @throws IOException
     * @throws IllegalArgumentException
     */
    public static void printWordVectors(String[] input) throws IOException, IllegalArgumentException {
        if (input.length != 2) {
            throw Usage.PRINT_WORD_VECTORS.toException();
        }
        Args args = createArgs();
        FastText fasttext = new FastText(args);
        fasttext.loadModel(input[1]);

        Scanner sc = new Scanner(System.in);
        while (sc.hasNextLine()) {
            String word = sc.nextLine();
            Vector vec = fasttext.getWordVector(word);
            System.out.println(word + " " + vec);
        }
    }

    /**
     * <pre>{@code void printSentenceVectors(const std::vector<std::string> args) {
     *  if (args.size() != 3) {
     *      printPrintSentenceVectorsUsage();
     *      exit(EXIT_FAILURE);
     *  }
     *  FastText fasttext;
     *  fasttext.loadModel(std::string(args[2]));
     *  Vector svec(fasttext.getDimension());
     *  while (std::cin.peek() != EOF) {
     *      fasttext.getSentenceVector(std::cin, svec);
     *      // Don't print sentence
     *      std::cout << svec << std::endl;
     *  }
     *  exit(0);
     * }}</pre>
     *
     * @param input
     * @throws IOException
     * @throws IllegalArgumentException
     */
    public static void printSentenceVectors(String[] input) throws IOException, IllegalArgumentException {
        if (input.length != 2) {
            throw Usage.PRINT_SENTENCE_VECTORS.toException();
        }
        Args args = createArgs();
        FastText fasttext = new FastText(args);
        fasttext.loadModel(input[1]);

        Scanner sc = new Scanner(System.in);
        while (sc.hasNextLine()) {
            Vector res = fasttext.getSentenceVector(sc.nextLine());
            System.out.println(res);
        }
    }

    /**
     * <pre>{@code void printNgrams(const std::vector<std::string> args) {
     *  if (args.size() != 4) {
     *      printPrintNgramsUsage();
     *      exit(EXIT_FAILURE);
     *  }
     *  FastText fasttext;
     *  fasttext.loadModel(std::string(args[2]));
     *  fasttext.ngramVectors(std::string(args[3]));
     *  exit(0);
     * }}</pre>
     *
     * @param input
     * @throws IOException
     * @throws IllegalArgumentException
     */
    public static void printNgrams(String[] input) throws IOException, IllegalArgumentException {
        if (input.length != 3) {
            throw Usage.PRINT_NGRAMS.toException();
        }
        Args args = createArgs();
        FastText fasttext = new FastText(args);
        fasttext.loadModel(input[1]);
        fasttext.ngramVectors(System.out, input[2]);
    }


    /**
     * <pre>{@code void train(const std::vector<std::string> args) {
     *  std::shared_ptr<Args> a = std::make_shared<Args>();
     *  a->parseArgs(args);
     *  FastText fasttext;
     *  fasttext.train(a);
     *  fasttext.saveModel();
     *  fasttext.saveVectors();
     *  if (a->saveOutput > 0) {
     *      fasttext.saveOutput();
     *  }
     * }}</pre>
     *
     * @param args
     * @throws Exception
     */
    public static void train(String[] args) throws Exception {
        new FastText(parseArgs(args)).trainAndSave();
    }

    public static void main(String... args) {
        try {
            run(args);
        } catch (IllegalArgumentException e) {
            System.out.print(e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void run(String... args) throws Exception {
        if (args.length < 1) {
            throw Usage.COMMON.toException();
        }
        String command = args[0];
        if ("skipgram".equalsIgnoreCase(command) || "cbow".equalsIgnoreCase(command) || "supervised".equalsIgnoreCase(command)) {
            train(args);
        } else if ("test".equalsIgnoreCase(command)) {
            test(args);
        } else if ("quantize".equalsIgnoreCase(command)) { // TODO: quantize
            throw new UnsupportedOperationException("TODO");
        } else if ("print-word-vectors".equalsIgnoreCase(command)) {
            printWordVectors(args);
        } else if ("print-sentence-vectors".equalsIgnoreCase(command)) {
            printSentenceVectors(args);
        } else if ("print-ngrams".equalsIgnoreCase(command)) {
            printNgrams(args);
        } else if ("nn".equalsIgnoreCase(command)) {// TODO: nn
            throw new UnsupportedOperationException("TODO");
        } else if ("analogies".equalsIgnoreCase(command)) {// TODO: analogies
            throw new UnsupportedOperationException("TODO");
        } else if ("predict".equalsIgnoreCase(command) || "predict-prob".equalsIgnoreCase(command)) {
            predict(args);
        } else {
            throw Usage.COMMON.toException();
        }
    }

    private enum Usage {
        COMMON("usage: {fasttext} <command> <args>\n\n"
                + "The commands supported by fasttext are:\n\n"
                + "  supervised              train a supervised classifier\n"
                + "  quantize                quantize a model to reduce the memory usage\n"
                + "  test                    evaluate a supervised classifier\n"
                + "  predict                 predict most likely labels\n"
                + "  predict-prob            predict most likely labels with probabilities\n"
                + "  skipgram                train a skipgram model\n"
                + "  cbow                    train a cbow model\n"
                + "  print-word-vectors      print word vectors given a trained model\n"
                + "  print-sentence-vectors  print sentence vectors given a trained model\n"
                + "  nn                      query for nearest neighbors\n"
                + "  analogies               query for analogies\n"),
        QUANTIZE("usage: {fasttext} quantize <args>"),
        TEST("usage: {fasttext} test <model> <test-data> [<k>]\n\n"
                + "  <model>      model filename\n"
                + "  <test-data>  test data filename (if -, read from stdin)\n"
                + "  <k>          (optional; 1 by default) predict top k labels\n"),
        PREDICT("usage: {fasttext} predict[-prob] <model> <test-data> [<k>]\n\n"
                + "  <model>      model filename\n"
                + "  <test-data>  test data filename (if -, read from stdin)\n"
                + "  <k>          (optional; 1 by default) predict top k labels\n"),
        PRINT_WORD_VECTORS("usage: {fasttext} print-word-vectors <model>\n\n"
                + "  <model>      model filename\n"),
        PRINT_SENTENCE_VECTORS("usage: {fasttext} print-sentence-vectors <model>\n\n"
                + "  <model>      model filename\n"),
        PRINT_NGRAMS("usage: {fasttext} print-ngrams <model> <word>\n\n"
                + "  <model>      model filename\n"
                + "  <word>       word to print\n"),
        NN("usage: {fasttext} nn <model> <k>\n\n"
                + "  <model>      model filename\n"
                + "  <k>          (optional; 10 by default) predict top k labels\n"),
        ANALOGIES("usage: {fasttext} analogies <model> <k>\n\n"
                + "  <model>      model filename\n"
                + "  <k>          (optional; 10 by default) predict top k labels\n"),;
        private final String message;

        Usage(String msg) {
            this.message = msg;
        }

        public String getMessage() {
            return message.replace("{fasttext}", "java -jar fasttext.jar");
        }

        public IllegalArgumentException toException() {
            return new IllegalArgumentException(getMessage());
        }
    }
}
