package cc.fasttext;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntConsumer;

import ru.avicomp.io.IOStreams;
import ru.avicomp.io.impl.LocalIOStreams;

/**
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.cc'>main.cc</a>
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/main.h'>main.h</a>
 */
public class Main {
    private static IOStreams fileSystem = new LocalIOStreams();

    public static void setFileSystem(IOStreams fileSystem) {
        Main.fileSystem = Objects.requireNonNull(fileSystem, "Null file system.");
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
        FastText fasttext = loadModel(input[1]);
        fasttext.setPrintOut(System.out);
        String infile = input[2];
        if ("-".equals(infile)) {
            fasttext.test(System.in, k);
            return;
        }
        if (!fasttext.ioStreams().canRead(infile)) {
            throw new IOException("Input file cannot be opened!");
        }
        try (InputStream in = fasttext.ioStreams().openInput(infile)) {
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
        FastText fasttext = loadModel(input[1]);
        fasttext.setPrintOut(System.out);
        String infile = input[2];
        if ("-".equals(infile)) { // read from pipe:
            fasttext.predict(System.in, k, printProb);
            return;
        }
        if (!fasttext.ioStreams().canRead(infile)) {
            throw new IOException("Input file cannot be opened!");
        }
        try (InputStream in = fasttext.ioStreams().openInput(infile)) {
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
        FastText fasttext = loadModel(input[1]);
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
        FastText fasttext = loadModel(input[1]);
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
        FastText fasttext = loadModel(input[1]);
        fasttext.ngramVectors(System.out, input[2]);
    }

    /**
     * <pre>{@code void nn(const std::vector<std::string> args) {
     *  int32_t k;
     *  if (args.size() == 3) {
     *      k = 10;
     *  } else if (args.size() == 4) {
     *      k = std::stoi(args[3]);
     *  } else {
     *      printNNUsage();
     *      exit(EXIT_FAILURE);
     *  }
     *  FastText fasttext;
     *  fasttext.loadModel(std::string(args[2]));
     *  fasttext.nn(k);
     *  exit(0);
     * }}</pre>
     *
     * @param input
     * @throws IOException
     * @throws IllegalArgumentException
     */
    public static void nn(String[] input) throws IOException, IllegalArgumentException {
        int k = 10;
        if (input.length == 3) {
            k = Integer.parseInt(input[2]);
        } else if (input.length != 2) {
            throw Usage.NN.toException();
        }
        FastText fasttext = loadModel(input[1]);
        fasttext.getPrecomputedWordVectors();
        Scanner sc = new Scanner(System.in);
        PrintStream out = System.out;
        while (true) {
            out.println("Query word?");
            String line;
            try {
                line = sc.next();
            } catch (NoSuchElementException e) {
                // ctrl+d
                return;
            }
            fasttext.nn(k, line).forEach((f, s) -> out.println(s + " " + Utils.formatNumber(f)));
        }
    }

    /**
     * <pre>{@code void analogies(const std::vector<std::string> args) {
     *  int32_t k;
     *  if (args.size() == 3) {
     *      k = 10;
     *  } else if (args.size() == 4) {
     *      k = std::stoi(args[3]);
     *  } else {
     *      printAnalogiesUsage();
     *      exit(EXIT_FAILURE);
     *  }
     *  FastText fasttext;
     *  fasttext.loadModel(std::string(args[2]));
     *  fasttext.analogies(k);
     *  exit(0);
     * }}</pre>
     *
     * @param input
     * @throws IOException
     * @throws IllegalArgumentException
     */
    public static void analogies(String[] input) throws IOException, IllegalArgumentException {
        int k = 10;
        if (input.length == 3) {
            k = Integer.parseInt(input[2]);
        } else if (input.length != 2) {
            throw Usage.ANALOGIES.toException();
        }
        FastText fasttext = loadModel(input[1]);
        fasttext.getPrecomputedWordVectors();
        Scanner sc = new Scanner(System.in);
        PrintStream out = System.out;
        while (true) {
            out.println("Query triplet (A - B + C)?");
            List<String> words = new ArrayList<>();
            while (words.size() < 3) {
                String word;
                try {
                    word = sc.next();
                } catch (NoSuchElementException e) {
                    // ctrl+d
                    return;
                }
                words.add(word);
            }
            fasttext.analogies(k, words.get(0), words.get(1), words.get(2)).forEach((f, s) -> out.println(s + " " + Utils.formatNumber(f)));
        }
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
     * @param input
     * @throws Exception
     */
    public static void train(String[] input) throws Exception {
        Args args = parseArgs(input);
        FastText fasttext = new FastText(args);
        fasttext.train();
        fasttext.saveModel();
        fasttext.saveVectors();
        if (args.saveOutput() > 0) {
            fasttext.saveOutput();
        }
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
        } else if ("nn".equalsIgnoreCase(command)) {
            nn(args);
        } else if ("analogies".equalsIgnoreCase(command)) {
            analogies(args);
        } else if ("predict".equalsIgnoreCase(command) || "predict-prob".equalsIgnoreCase(command)) {
            predict(args);
        } else {
            throw Usage.COMMON.toException();
        }
    }

    /**
     * A factory method to load new {@link FastText model}.
     *
     * @param file, String, not null, the reference to file
     * @return {@link FastText}
     * @throws IOException if something is wrong.
     */
    public static FastText loadModel(String file) throws IOException {
        return FastText.loadModel(fileSystem, file);
    }

    /**
     * Creates an empty Args
     *
     * @return {@link Args}
     */
    public static Args createArgs() {
        return new Args.Builder().build();
    }

    /**
     * from args.cc:
     * <pre>{@code void Args::parseArgs(const std::vector<std::string>& args) {
     *  std::string command(args[1]);
     *  if (command == "supervised") {
     *      model = model_name::sup;
     *      loss = loss_name::softmax;
     *      minCount = 1;
     *      minn = 0;
     *      maxn = 0;
     *      lr = 0.1;
     *  } else if (command == "cbow") {
     *      model = model_name::cbow;
     *  }
     *  int ai = 2;
     *  while (ai < args.size()) {
     *      if (args[ai][0] != '-') {
     *          std::cerr << "Provided argument without a dash! Usage:" << std::endl;
     *          printHelp();
     *          exit(EXIT_FAILURE);
     *      }
     *      if (args[ai] == "-h") {
     *          std::cerr << "Here is the help! Usage:" << std::endl;
     *          printHelp();
     *          exit(EXIT_FAILURE);
     *      } else if (args[ai] == "-input") {
     *          input = std::string(args[ai + 1]);
     *      } else if (args[ai] == "-test") {
     *          test = std::string(args[ai + 1]);
     *      } else if (args[ai] == "-output") {
     *          output = std::string(args[ai + 1]);
     *      } else if (args[ai] == "-lr") {
     *          lr = std::stof(args[ai + 1]);
     *      } else if (args[ai] == "-lrUpdateRate") {
     *          lrUpdateRate = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-dim") {
     *          dim = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-ws") {
     *          ws = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-epoch") {
     *          epoch = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-minCount") {
     *          minCount = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-minCountLabel") {
     *          minCountLabel = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-neg") {
     *          neg = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-wordNgrams") {
     *          wordNgrams = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-loss") {
     *          if (args[ai + 1] == "hs") {
     *              loss = loss_name::hs;
     *          } else if (args[ai + 1] == "ns") {
     *              loss = loss_name::ns;
     *          } else if (args[ai + 1] == "softmax") {
     *              loss = loss_name::softmax;
     *          } else {
     *              std::cerr << "Unknown loss: " << args[ai + 1] << std::endl;
     *              printHelp();
     *              exit(EXIT_FAILURE);
     *          }
     *      } else if (args[ai] == "-bucket") {
     *          bucket = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-minn") {
     *          minn = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-maxn") {
     *          maxn = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-thread") {
     *          thread = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-t") {
     *          t = std::stof(args[ai + 1]);
     *      } else if (args[ai] == "-label") {
     *          label = std::string(args[ai + 1]);
     *      } else if (args[ai] == "-verbose") {
     *          verbose = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-pretrainedVectors") {
     *          pretrainedVectors = std::string(args[ai + 1]);
     *      } else if (args[ai] == "-saveOutput") {
     *          saveOutput = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-qnorm") {
     *          qnorm = true; ai--;
     *      } else if (args[ai] == "-retrain") {
     *          retrain = true; ai--;
     *      } else if (args[ai] == "-qout") {
     *          qout = true; ai--;
     *      } else if (args[ai] == "-cutoff") {
     *          cutoff = std::stoi(args[ai + 1]);
     *      } else if (args[ai] == "-dsub") {
     *          dsub = std::stoi(args[ai + 1]);
     *      } else {
     *          std::cerr << "Unknown argument: " << args[ai] << std::endl;
     *          printHelp();
     *          exit(EXIT_FAILURE);
     *      }
     *      ai += 2;
     *  }
     *  if (input.empty() || output.empty()) {
     *      std::cerr << "Empty input or output path." << std::endl;
     *      printHelp();
     *      exit(EXIT_FAILURE);
     *  }
     *  if (wordNgrams <= 1 && maxn == 0) {
     *      bucket = 0;
     *  }
     * }}</pre>
     *
     * @param args array of strings, not null, not empty
     * @return {@link Args} object
     * @throws IllegalArgumentException if wrong arguments in input
     */
    public static Args parseArgs(String... args) throws IllegalArgumentException {
        if (args.length == 0) {
            throw Usage.ARGS.toException("Empty args specified");
        }
        Args.Builder builder = new Args.Builder();
        builder.setModel(Args.ModelName.fromName(args[0]));
        Map<String, String> map = toMap(args);
        if (map.containsKey("-h")) {
            throw Usage.ARGS.toException("Here is the help! Usage:");
        }
        putIntegerArg(map, "-lrUpdateRate", builder::setLRUpdateRate);
        putIntegerArg(map, "-dim", builder::setDim);
        putIntegerArg(map, "-ws", builder::setWS);
        putIntegerArg(map, "-epoch", builder::setEpoch);
        putIntegerArg(map, "-minCount", builder::setMinCount);
        putIntegerArg(map, "-minCountLabel", builder::setMinCountLabel);
        putIntegerArg(map, "-neg", builder::setNeg);
        putIntegerArg(map, "-wordNgrams", builder::setWordNgrams);
        putIntegerArg(map, "-bucket", builder::setBucket);
        putIntegerArg(map, "-minn", builder::setMinN);
        putIntegerArg(map, "-maxn", builder::setMaxN);
        putIntegerArg(map, "-thread", builder::setThread);
        putIntegerArg(map, "-verbose", builder::setVerbose);
        putIntegerArg(map, "-saveOutput", builder::setSaveOutput);
        putIntegerArg(map, "-cutoff", builder::setCutOff);
        putIntegerArg(map, "-dsub", builder::setDSub);

        putDoubleArg(map, "-lr", builder::setLR);
        putDoubleArg(map, "-t", builder::setSamplingThreshold);

        putBooleanArg(map, "-qnorm", builder::setQNorm);
        putBooleanArg(map, "-retrain", builder::setRetrain);
        putBooleanArg(map, "-qout", builder::setQOut);

        putStringArg(map, "-pretrainedVectors", builder::setPreparedVectors);
        putStringArg(map, "-label", builder::setLabel);

        if (map.containsKey("-loss")) {
            builder.setLossName(Args.LossName.fromName(map.get("-loss")));
        }

        Args res = builder.build();
        // todo: temporary - should not be in args
        res.input = map.get("-input");
        res.output = map.get("-output");
        return res;
    }

    /**
     * Parses an array to Map
     * Example: "cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5 -input %s -output %s" =>
     * "[cbow=null, -thread=4, -dim=128, -ws=5, -epoch=10, -minCount=5, -input=%s, -output=%s]"
     *
     * @param input array of strings
     * @return Map
     */
    public static Map<String, String> toMap(String... input) {
        Map<String, String> res = new LinkedHashMap<>();
        for (int i = 0; i < input.length; i++) {
            if (input[i].startsWith("-")) {
                String key = input[i];
                String val = i == input.length - 1 ? null : input[++i];
                res.put(key, val == null || val.startsWith("-") ? Boolean.TRUE.toString() : val);
            } else {
                res.put(input[i], null);
            }
        }
        return res;
    }

    private static void putStringArg(Map<String, String> map, String key, Consumer<String> setter) {
        if (!map.containsKey(key)) return;
        setter.accept(Objects.requireNonNull(map.get(key), "Null value for " + key));
    }

    private static void putIntegerArg(Map<String, String> map, String key, IntConsumer setter) {
        if (!map.containsKey(key)) return;
        String value = Objects.requireNonNull(map.get(key), "Null int value for " + key);
        try {
            setter.accept(Integer.parseInt(value));
        } catch (NumberFormatException n) {
            throw Usage.ARGS.toException("Wrong value for " + key + ": " + n.getMessage());
        }
    }

    private static void putDoubleArg(Map<String, String> map, String key, DoubleConsumer setter) {
        if (!map.containsKey(key)) return;
        String value = Objects.requireNonNull(map.get(key), "Null double value for " + key);
        try {
            setter.accept(Double.parseDouble(value));
        } catch (NumberFormatException n) {
            throw Usage.ARGS.toException("Wrong value for " + key + ": " + n.getMessage());
        }
    }

    private static void putBooleanArg(Map<String, String> map, String key, Consumer<Boolean> setter) {
        if (!map.containsKey(key)) return;
        String value = Objects.requireNonNull(map.get(key), "Null value for " + key);
        setter.accept(Boolean.parseBoolean(value));
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
                + "  <k>          (optional; 10 by default) predict top k labels\n"),

        ARGS_BASIC_HELP("\nThe following arguments are mandatory:\n"
                + "  -input              training file path\n"
                + "  -output             output file path\n"
                + "\nThe following arguments are optional:\n"
                + "  -verbose            verbosity level [integer]\n"),
        ARGS_DICTIONARY_HELP("\nThe following arguments for the dictionary are optional:\n"
                + "  -minCount           minimal number of word occurences [integer]\n"
                + "  -minCountLabel      minimal number of label occurences [integer]\n"
                + "  -wordNgrams         max length of word ngram [integer]\n"
                + "  -bucket             number of buckets [integer]\n"
                + "  -minn               min length of char ngram [integer]\n"
                + "  -maxn               max length of char ngram [integer]\n"
                + "  -t                  sampling threshold [double]\n"
                + "  -label              labels prefix [string]\n"),
        ARGS_TRAINING_HELP("\nThe following arguments for training are optional:\n"
                + "  -lr                 learning rate [double]\n"
                + "  -lrUpdateRate       change the rate of updates for the learning rate [integer]\n"
                + "  -dim                size of word vectors [integer]\n"
                + "  -ws                 size of the context window [integer]\n"
                + "  -epoch              number of epochs [integer]\n"
                + "  -neg                number of negatives sampled [integer]\n"
                + "  -loss               loss function {ns, hs, softmax} [enum]\n"
                + "  -thread             number of threads [integer]\n"
                + "  -pretrainedVectors  pretrained word vectors for supervised learning [string]\n"
                + "  -saveOutput         whether output params should be saved [integer]\n"),
        ARGS_QUANTIZATION_HELP("\nThe following arguments for quantization are optional:\n"
                + "  -cutoff             number of words and ngrams to retain [integer]\n"
                + "  -retrain            finetune embeddings if a cutoff is applied [boolean]\n"
                + "  -qnorm              quantizing the norm separately [boolean]\n"
                + "  -qout               quantizing the classifier [boolean\n"
                + "  -dsub               size of each sub-vector [integer]\n"),
        ARGS(ARGS_BASIC_HELP.message + ARGS_DICTIONARY_HELP.message + ARGS_TRAINING_HELP.message + ARGS_QUANTIZATION_HELP.message);

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

        public IllegalArgumentException toException(String line) {
            return new IllegalArgumentException(line + "\n" + getMessage());
        }
    }
}
