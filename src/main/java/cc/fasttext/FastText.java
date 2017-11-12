package cc.fasttext;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.UniformIntegerDistribution;

import cc.fasttext.Args.ModelName;
import cc.fasttext.Dictionary.EntryType;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import fasttext.io.BufferedLineReader;
import fasttext.io.LineReader;
import ru.avicomp.io.FTInputStream;
import ru.avicomp.io.FTOutputStream;
import ru.avicomp.io.FTReader;

/**
 * FastText class, can be used as a lib in other projects
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc'>fasttext.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/fasttext.h'>fasttext.h</a>
 *
 * @author Ivan
 */
public strictfp class FastText {

    public static final int FASTTEXT_VERSION = 12;
    public static final int FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

    private final Args args_;
    private Dictionary dict_;
    private Matrix input_;
    private QMatrix qinput_;
    private Matrix output_;
    private QMatrix qoutput_;
    private Model model_;

    private AtomicLong tokenCount_;
    private long start_;
    private boolean quant_;
    private int version;

    private PrintStream out = System.out;

    private Class<? extends LineReader> lineReaderClass_ = BufferedLineReader.class;
    private long threadFileSize;

    public FastText(Args args) {
        this.args_ = args;
    }

    /**
     * Sets output to log.
     *
     * @param out {@link PrintStream} set output to log
     */
    public void setPrintOut(PrintStream out) {
        this.out = Objects.requireNonNull(out, "Null out");
    }

    /**
     * <pre>{@code
     * void FastText::getVector(Vector& vec, const std::string& word) const {
     *  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
     *  vec.zero();
     *  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
     *      if (quant_) {
     *          vec.addRow(*qinput_, *it);
     *      } else {
     *          vec.addRow(*input_, *it);
     *      }
     *  }
     *  if (ngrams.size() > 0) {
     *      vec.mul(1.0 / ngrams.size());
     *  }
     * }
     * }</pre>
     *
     * @param word
     * @return
     */
    public Vector getVector(String word) {
        Vector res = new Vector(args_.dim);
        List<Integer> ngrams = dict_.getSubwords(word);
        for (Integer it : ngrams) {
            if (quant_) {
                res.addRow(qinput_, it);
            } else {
                res.addRow(input_, it);
            }
        }
        if (ngrams.size() > 0) {
            res.mul(1.0f / ngrams.size());
        }
        return res;
    }

    /**
     * <pre>{@code
     * void FastText::saveVectors() {
     *  std::ofstream ofs(args_->output + ".vec");
     *  if (!ofs.is_open()) {
     *      std::cerr << "Error opening file for saving vectors." << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     *  ofs << dict_->nwords() << " " << args_->dim << std::endl;
     *  Vector vec(args_->dim);
     *  for (int32_t i = 0; i < dict_->nwords(); i++) {
     *      std::string word = dict_->getWord(i);
     *      getVector(vec, word);
     *      ofs << word << " " << vec << std::endl;
     *  }
     *  ofs.close();
     * }
     * }</pre>
     *
     * @throws IOException
     */
    public void saveVectors() throws IOException {
        if (Utils.isEmpty(args_.output)) {
            if (args_.verbose > 1) {
                out.println("output is empty, skip save vector file");
            }
            return;
        }
        // validate and prepare:
        Path file = Paths.get(args_.output + ".vec");
        args_.getIOStreams().prepareParent(file.toString());
        if (!args_.getIOStreams().canWrite(file.toString())) {
            throw new IOException("Can't write to " + file);
        }
        if (args_.verbose > 1) {
            out.println("Saving Vectors to " + file.toAbsolutePath());
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(args_.getIOStreams().createOutput(file.toString()), args_.charset))) {
            writer.write(dict_.nwords() + " " + args_.dim + "\n");
            for (int i = 0; i < dict_.nwords(); i++) {
                String word = dict_.getWord(i);
                Vector vec = getVector(word);
                writer.write(word);
                for (int j = 0; j < vec.m_; j++) {
                    writer.write(" ");
                    writer.write(Utils.formatNumber(vec.data_[j]));
                }
                writer.write("\n");
            }
            writer.flush();
        }
    }

    public void saveOutput() {
        // TODO:
    }

    /**
     * Checks model versions.
     * <pre>{@code bool FastText::checkModel(std::istream& in) {
     *  int32_t magic;
     *  in.read((char*)&(magic), sizeof(int32_t));
     *  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
     *      return false;
     *  }
     *  in.read((char*)&(version), sizeof(int32_t));
     *  if (version > FASTTEXT_VERSION) {
     *      return false;
     *  }
     *  return true;
     * }}</pre>
     *
     * @param in {@link FTInputStream} binary just opened input stream
     * @return true if version is okay
     * @throws IOException if something is wrong
     */
    private boolean checkModel(FTInputStream in) throws IOException {
        int magic = in.readInt();
        int version = in.readInt();
        return FASTTEXT_FILEFORMAT_MAGIC_INT32 == magic && (this.version = version) == FASTTEXT_VERSION;
    }

    /**
     * Writes versions to the model.
     * <pre>{@code
     * void FastText::signModel(std::ostream& out) {
     *  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
     *  const int32_t version = FASTTEXT_VERSION;
     *  out.write((char*)&(magic), sizeof(int32_t));
     *  out.write((char*)&(version), sizeof(int32_t));
     * }}</pre>
     *
     * @param out {@link FTOutputStream} binary just opened output stream
     * @throws IOException if something is wrong
     */
    private void signModel(FTOutputStream out) throws IOException {
        out.writeInt(FASTTEXT_FILEFORMAT_MAGIC_INT32);
        out.writeInt(FASTTEXT_VERSION);
    }


    /**
     * <pre>{@code
     * void FastText::saveModel() {
     *  std::string fn(args_->output);
     *  if (quant_) {
     *      fn += ".ftz";
     *  } else {
     *      fn += ".bin";
     *  }
     *  std::ofstream ofs(fn, std::ofstream::binary);
     *  if (!ofs.is_open()) {
     *      std::cerr << "Model file cannot be opened for saving!" << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     *  signModel(ofs);
     *  args_->save(ofs);
     *  dict_->save(ofs);
     *
     *  ofs.write((char*)&(quant_), sizeof(bool));
     *  if (quant_) {
     *      qinput_->save(ofs);
     *  } else {
     *      input_->save(ofs);
     *  }
     *  ofs.write((char*)&(args_->qout), sizeof(bool));
     *  if (quant_ && args_->qout) {
     *      qoutput_->save(ofs);
     *  } else {
     *      output_->save(ofs);
     *  }
     *  ofs.close();
     * }
     * }</pre>
     *
     * @throws IOException
     */
    public void saveModel() throws IOException {
        if (Utils.isEmpty(args_.output)) {
            if (args_.verbose > 1) {
                out.println("output is empty, skip save model file");
            }
            return;
        }

        // validate and prepare (todo: move to the beginning):
        Path file = Paths.get(args_.output + (quant_ ? ".ftz" : ".bin"));
        args_.getIOStreams().prepareParent(file.toString());
        if (!args_.getIOStreams().canWrite(file.toString())) {
            throw new IOException("Can't write to " + file);
        }
        if (args_.verbose > 1) {
            out.println("Saving model to " + file.toAbsolutePath());
        }

        try (FTOutputStream out = new FTOutputStream(new BufferedOutputStream(args_.getIOStreams().createOutput(file.toString())))) {
            signModel(out);
            args_.save(out);
            dict_.save(out);

            out.writeBoolean(quant_);
            if (quant_) {
                qinput_.save(out);
            } else {
                input_.save(out);
            }

            out.writeBoolean(args_.qout);
            if (quant_ && args_.qout) {
                qinput_.save(out);
            } else {
                output_.save(out);
            }
        }
    }

    /**
     * <pre>{@code
     * void FastText::loadModel(const std::string& filename) {
     *  std::ifstream ifs(filename, std::ifstream::binary);
     *  if (!ifs.is_open()) {
     *      std::cerr << "Model file cannot be opened for loading!" << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     *  if (!checkModel(ifs)) {
     *      std::cerr << "Model file has wrong file format!" << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     *  loadModel(ifs);
     *  ifs.close();
     * }}</pre>
     *
     * @param file, String, path to model
     * @throws IOException              if an I/O error occurs
     * @throws IllegalArgumentException if model is wrong
     */
    public void loadModel(String file) throws IOException, IllegalArgumentException {
        Path path = Paths.get(file);
        if (!args_.getIOStreams().canRead(path.toString())) {
            throw new IOException("Model file cannot be opened for loading: <" + path.toAbsolutePath() + ">");
        }
        try (FTInputStream in = new FTInputStream(new BufferedInputStream(args_.getIOStreams().openInput(path.toString())))) {
            if (!checkModel(in)) throw new IllegalArgumentException("Model file has wrong format!");
            loadModel(in);
        }
    }

    /**
     * <pre>{@code
     * void FastText::loadModel(std::istream& in) {
     *  args_ = std::make_shared<Args>();
     *  dict_ = std::make_shared<Dictionary>(args_);
     *  input_ = std::make_shared<Matrix>();
     *  output_ = std::make_shared<Matrix>();
     *  qinput_ = std::make_shared<QMatrix>();
     *  qoutput_ = std::make_shared<QMatrix>();
     *  args_->load(in);
     *  if (version == 11 && args_->model == model_name::sup) {
     *      // backward compatibility: old supervised models do not use char ngrams.
     *      args_->maxn = 0;
     *  }
     *  dict_->load(in);
     *  bool quant_input;
     *  in.read((char*) &quant_input, sizeof(bool));
     *  if (quant_input) {
     *      quant_ = true;
     *      qinput_->load(in);
     *  } else {
     *      input_->load(in);
     *  }
     *  if (!quant_input && dict_->isPruned()) {
     *      std::cerr << "Invalid model file.\n" << "Please download the updated model from www.fasttext.cc.\n" << "See issue #332 on Github for more information.\n";
     *      exit(1);
     *  }
     *  in.read((char*) &args_->qout, sizeof(bool));
     *  if (quant_ && args_->qout) {
     *      qoutput_->load(in);
     *  } else {
     *      output_->load(in);
     *  }
     *  model_ = std::make_shared<Model>(input_, output_, args_, 0);
     *  model_->quant_ = quant_;
     *  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);
     *  if (args_->model == model_name::sup) {
     *      model_->setTargetCounts(dict_->getCounts(entry_type::label));
     *  } else {
     *      model_->setTargetCounts(dict_->getCounts(entry_type::word));
     *  }
     * }}</pre>
     *
     * @param in {@link FTInputStream}
     * @throws IOException              io-error
     * @throws IllegalArgumentException if wrong input
     */
    private void loadModel(FTInputStream in) throws IOException {
        args_.load(in);
        input_ = new Matrix();
        output_ = new Matrix();
        qinput_ = new QMatrix();
        qoutput_ = new QMatrix();
        if (version == 11 && args_.model == ModelName.SUP) {
            // backward compatibility: old supervised models do not use char ngrams.
            args_.maxn = 0;
        }
        dict_ = new Dictionary(args_);
        dict_.load(in);
        boolean quant_input = in.readBoolean();
        if (quant_input) {
            quant_ = true;
            qinput_.load(in);
        } else {
            input_.load(in);
        }

        if (!quant_input && dict_.isPruned()) {
            throw new IllegalArgumentException("Invalid model file.\n" +
                    "Please download the updated model from www.fasttext.cc.\n" +
                    "See issue #332 on Github for more information.\n");
        }
        args_.qout = in.readBoolean();
        if (quant_ && args_.qout) {
            qoutput_.load(in);
        } else {
            output_.load(in);
        }

        model_ = new Model(input_, output_, args_, 0);
        model_.quant_ = quant_;
        model_.setQuantizePointer(qinput_, qoutput_, args_.qout);
        if (args_.model == Args.ModelName.SUP) {
            model_.setTargetCounts(dict_.getCounts(EntryType.LABEL));
        } else {
            model_.setTargetCounts(dict_.getCounts(EntryType.WORD));
        }
    }

    public void printInfo(float progress, float loss) {
        float t = (float) (System.currentTimeMillis() - start_) / 1000;
        float ws = (float) (tokenCount_.get()) / t;
        float wst = (float) (tokenCount_.get()) / t / args_.thread;
        float lr = (float) (args_.lr * (1.0f - progress));
        int eta = (int) (t / progress * (1 - progress));
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
        out.printf("\rProgress: %.1f%% words/sec: %d words/sec/thread: %d lr: %.6f loss: %.6f eta: %d h %d m",
                100 * progress, (int) ws, (int) wst, lr, loss, etah, etam);
    }

    /**
     * <pre>{@code
     * void FastText::supervised(Model& model, real lr, const std::vector<int32_t>& line, const std::vector<int32_t>& labels) {
     *  if (labels.size() == 0 || line.size() == 0) return;
     *  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
     *  int32_t i = uniform(model.rng);
     *  model.update(line, labels[i], lr);
     * }
     * }</pre>
     *
     * @param model
     * @param lr
     * @param line
     * @param labels
     */
    public void supervised(Model model, float lr, final List<Integer> line, final List<Integer> labels) {
        if (labels.size() == 0 || line.size() == 0)
            return;
        UniformIntegerDistribution uniform = new UniformIntegerDistribution(model.rng, 0, labels.size() - 1);
        int i = uniform.sample();
        model.update(line, labels.get(i), lr);
    }

    /**
     * <pre>{@code
     * void FastText::cbow(Model& model, real lr, const std::vector<int32_t>& line) {
     *  std::vector<int32_t> bow;
     *  std::uniform_int_distribution<> uniform(1, args_->ws);
     *  for (int32_t w = 0; w < line.size(); w++) {
     *      int32_t boundary = uniform(model.rng);
     *      bow.clear();
     *      for (int32_t c = -boundary; c <= boundary; c++) {
     *          if (c != 0 && w + c >= 0 && w + c < line.size()) {
     *              const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
     *              bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
     *          }
     *      }
     *      model.update(bow, line[w], lr);
     *  }
     * }
     * }</pre>
     *
     * @param model
     * @param lr
     * @param line
     */
    public void cbow(Model model, float lr, final List<Integer> line) {
        List<Integer> bow = new ArrayList<>();
        UniformIntegerDistribution uniform = new UniformIntegerDistribution(model.rng, 1, args_.ws);
        for (int w = 0; w < line.size(); w++) {
            bow.clear(); // don't create new one to work with encapsulated big array inside list
            int boundary = uniform.sample();
            for (int c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    List<Integer> ngrams = dict_.getSubwords(line.get(w + c));
                    bow.addAll(ngrams);
                }
            }
            model.update(bow, line.get(w), lr);
        }
    }

    /**
     * <pre>{@code
     * void FastText::skipgram(Model& model, real lr, const std::vector<int32_t>& line) {
     *  std::uniform_int_distribution<> uniform(1, args_->ws);
     *  for (int32_t w = 0; w < line.size(); w++) {
     *      int32_t boundary = uniform(model.rng);
     *      const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
     *      for (int32_t c = -boundary; c <= boundary; c++) {
     *          if (c != 0 && w + c >= 0 && w + c < line.size()) {
     *              model.update(ngrams, line[w + c], lr);
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param model
     * @param lr
     * @param line
     */
    public void skipgram(Model model, float lr, final List<Integer> line) {
        UniformIntegerDistribution uniform = new UniformIntegerDistribution(model.rng, 1, args_.ws);
        for (int w = 0; w < line.size(); w++) {
            int boundary = uniform.sample();
            List<Integer> ngrams = dict_.getSubwords(line.get(w));
            for (int c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model.update(ngrams, line.get(w + c), lr);
                }
            }
        }
    }

    public void test(InputStream in, int k) throws IOException, Exception {
        int nexamples = 0, nlabels = 0;
        double precision = 0.0f;
        List<Integer> line = new ArrayList<Integer>();
        List<Integer> labels = new ArrayList<Integer>();

        LineReader lineReader = null;
        try {
            lineReader = lineReaderClass_.getConstructor(InputStream.class, String.class).newInstance(in, args_.charset.name());
            String[] lineTokens;
            while ((lineTokens = lineReader.readLineTokens()) != null) {
                if (lineTokens.length == 1 && "quit".equals(lineTokens[0])) {
                    break;
                }
                dict_.getLine(lineTokens, line, labels, model_.rng);
                dict_.addNgrams(line, args_.wordNgrams);
                if (labels.size() > 0 && line.size() > 0) {
                    List<Pair<Float, Integer>> modelPredictions = new ArrayList<Pair<Float, Integer>>();
                    model_.predict(line, k, modelPredictions);
                    for (Pair<Float, Integer> pair : modelPredictions) {
                        if (labels.contains(pair.second())) {
                            precision += 1.0f;
                        }
                    }
                    nexamples++;
                    nlabels += labels.size();
                    // } else {
                    // out.println("FAIL Test line: " + lineTokens +
                    // "labels: " + labels + " line: " + line);
                }
            }
        } finally {
            if (lineReader != null) {
                lineReader.close();
            }
        }

        out.printf("P@%d: %.3f%n", k, precision / (k * nexamples));
        out.printf("R@%d: %.3f%n", k, precision / nlabels);
        out.println("Number of examples: " + nexamples);
    }

    /**
     * <pre>{@code
     * void FastText::predict(std::istream& in, int32_t k, std::vector<std::pair<real,std::string>>& predictions) const {
     *  std::vector<int32_t> words, labels;
     *  predictions.clear();
     *  dict_->getLine(in, words, labels, model_->rng);
     *  predictions.clear();
     *  if (words.empty()) return;
     *  Vector hidden(args_->dim);
     *  Vector output(dict_->nlabels());
     *  std::vector<std::pair<real,int32_t>> modelPredictions;
     *  model_->predict(words, k, modelPredictions, hidden, output);
     *  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
     *      predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
     *  }
     * }}</pre>
     *
     * @param in {@link FTReader}
     * @param k  number of labels for each input line, the size of multimap.
     * @return {@link Multimap}, labels as keys, probabilities (floats) as values
     * @throws IOException if something wrong
     */
    private Multimap<String, Float> predict(FTReader in, int k) throws IOException {
        List<Integer> words = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        dict_.getLine(in, words, labels);
        if (words.isEmpty()) {
            return ImmutableListMultimap.of();
        }
        Vector hidden = new Vector(args_.dim);
        Vector output = new Vector(dict_.nlabels());
        TreeMultimap<Float, Integer> map = model_._predict(words, k, hidden, output);
        @SuppressWarnings("ConstantConditions")
        Multimap<String, Float> res = TreeMultimap.create(this::compareLabels, map.keySet().comparator());
        map.forEach((f, i) -> res.put(dict_.getLabel(i), f));
        return res;
    }

    /**
     * Compares labels for output.
     *
     * @param left  String, label
     * @param right String, label
     * @return int
     */
    private int compareLabels(String left, String right) {
        String dig1, dig2;
        if ((dig1 = left.replace(args_.label, "")).matches("\\d+") && (dig2 = right.replace(args_.label, "")).matches("\\d+")) {
            return Integer.compare(Integer.valueOf(dig1), Integer.valueOf(dig2));
        }
        return left.compareTo(right);
    }

    /**
     * Predicts most likely labels.
     * Original code:
     * <pre>{@code void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
     *  std::vector<std::pair<real,std::string>> predictions;
     *  while (in.peek() != EOF) {
     *      predictions.clear();
     *      predict(in, k, predictions);
     *      if (predictions.empty()) {
     *          std::cout << std::endl;
     *          continue;
     *      }
     *      for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
     *          if (it != predictions.cbegin()) {
     *              std::cout << " ";
     *          }
     *          std::cout << it->second;
     *          if (print_prob) {
     *              std::cout << " " << exp(it->first);
     *          }
     *      }
     *      std::cout << std::endl;
     *  }
     * }}</pre>
     *
     * @param in        {@link InputStream} to read data
     * @param out       {@link PrintStream} to write data
     * @param k         the number of result labels
     * @param printProb if true include also probabilities to output
     * @throws IOException if something wrong.
     */
    public void predict(InputStream in, PrintStream out, int k, boolean printProb) throws IOException {
        Objects.requireNonNull(in, "Null input");
        Objects.requireNonNull(out, "Null output");
        if (k <= 0) throw new IllegalArgumentException("Negative factor");
        FTReader reader = new FTReader(in, args_.getCharset());
        while (!reader.end()) {
            Multimap<String, Float> predictions = predict(reader, k);
            if (predictions.isEmpty()) continue;
            String line = predictions.entries().stream().map(pair -> {
                String res = pair.getKey();
                if (printProb) {
                    res += " " + Utils.formatNumber(Math.exp(pair.getValue()));
                }
                return res;
            }).collect(Collectors.joining(" "));
            out.println(line);
        }
    }

    /**
     * Predicts and prints to standard output.
     *
     * @param in        {@link InputStream} to read data
     * @param k,        int the factor.
     * @param printProb to print probs.
     * @throws IOException if something is wrong.
     */
    public void predict(InputStream in, int k, boolean printProb) throws IOException {
        predict(in, out, k, printProb);
    }

    /**
     * Predicts the given line.
     *
     * @param line, String data to analyze
     * @param k,    int, the factor.
     * @return {@link Multimap}, labels as keys, probability as values
     * @throws IOException if something wrong.
     */
    public Multimap<String, Float> predict(String line, int k) throws IOException {
        InputStream in = new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8.name()));
        FTReader r = new FTReader(in, args_.getCharset());
        return predict(r, k);
    }

    public void wordVectors() {
        LineReader lineReader = null;
        try {
            lineReader = lineReaderClass_.getConstructor(InputStream.class, String.class).newInstance(System.in,
                    args_.charset.name());
            String word;
            while (!Utils.isEmpty((word = lineReader.readLine()))) {
                Vector vec = getVector(word);
                out.println(word + " " + vec);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (lineReader != null) {
                try {
                    lineReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public void textVectors() {
        List<Integer> line = new ArrayList<Integer>();
        List<Integer> labels = new ArrayList<Integer>();
        Vector vec = new Vector(args_.dim);
        LineReader lineReader = null;
        try {
            lineReader = lineReaderClass_.getConstructor(InputStream.class, String.class).newInstance(System.in, args_.charset.name());
            String[] lineTokens;
            while ((lineTokens = lineReader.readLineTokens()) != null) {
                if (lineTokens.length == 1 && "quit".equals(lineTokens[0])) {
                    break;
                }
                dict_.getLine(lineTokens, line, labels, model_.rng);
                dict_.addNgrams(line, args_.wordNgrams);
                vec.zero();
                for (Integer it : line) {
                    vec.addRow(input_, it);
                }
                if (!line.isEmpty()) {
                    vec.mul(1.0f / line.size());
                }
                out.println(vec);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (lineReader != null) {
                try {
                    lineReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public void printVectors() {
        if (args_.model == Args.ModelName.SUP) {
            textVectors();
        } else {
            wordVectors();
        }
    }

    private void loadVectors(String filename) throws IOException {
        List<String> words;
        Matrix mat; // temp. matrix for pretrained vectors
        int n, dim;

        BufferedReader dis = null;
        String line;
        String[] lineParts;
        try {
            dis = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));

            line = dis.readLine();
            lineParts = line.split(" ");
            n = Integer.parseInt(lineParts[0]);
            dim = Integer.parseInt(lineParts[1]);

            words = new ArrayList<>(n);

            if (dim != args_.dim) {
                throw new IllegalArgumentException(String.format("Dimension of pretrained vectors does not match args " +
                        "-dim option, pretrain dim is %d, args dim is %d", dim, args_.dim));
            }

            mat = new Matrix(n, dim);
            for (int i = 0; i < n; i++) {
                line = dis.readLine();
                lineParts = line.split(" ");
                String word = lineParts[0];
                for (int j = 1; j <= dim; j++) {
                    mat.data_[i][j - 1] = Float.parseFloat(lineParts[j]);
                }
                words.add(word);
                dict_.add(word);
            }

            dict_.threshold(1, 0);
            input_ = new Matrix(dict_.nwords() + args_.bucket, args_.dim);
            input_.uniform(args_.getRandomFactory().apply(1), 1.0f / args_.dim);
            for (int i = 0; i < n; i++) {
                int idx = dict_.getId(words.get(i));
                if (idx < 0 || idx >= dict_.nwords())
                    continue;
                for (int j = 0; j < dim; j++) {
                    input_.data_[idx][j] = mat.data_[i][j];
                }
            }

        } catch (IOException e) {
            throw new IOException("Pretrained vectors file cannot be opened!", e);
        } finally {
            try {
                if (dis != null) {
                    dis.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Trains.
     *
     * @throws IOException
     * @throws ExecutionException
     */
    public void train() throws IOException, ExecutionException {
        dict_ = new Dictionary(args_);

        if ("-".equals(args_.input)) {
            throw new IOException("Cannot use stdin for training!");
        }

        if (!args_.getIOStreams().canRead(args_.input)) {
            throw new IOException("Input file cannot be opened! " + args_.input);
        }
        dict_.readFromFile(args_);
        try (FTReader r = args_.createReader()) {
            threadFileSize = r.size();
        }

        if (!Utils.isEmpty(args_.pretrainedVectors)) {
            loadVectors(args_.pretrainedVectors);
        } else {
            input_ = new Matrix(dict_.nwords() + args_.bucket, args_.dim);
            input_.uniform(args_.getRandomFactory().apply(1), 1.0f / args_.dim);
        }

        if (args_.model == Args.ModelName.SUP) {
            output_ = new Matrix(dict_.nlabels(), args_.dim);
        } else {
            output_ = new Matrix(dict_.nwords(), args_.dim);
        }

        start_ = System.currentTimeMillis();
        tokenCount_ = new AtomicLong(0);
        if (args_.thread > 1) {
            ExecutorService service = //Executors.newFixedThreadPool(args_.thread);
                    Executors.newFixedThreadPool(args_.thread,
                            r -> {
                                Thread t = Executors.defaultThreadFactory().newThread(r);
                                t.setDaemon(true);
                                return t;
                            });
            Set<Future<Integer>> res = IntStream.range(0, args_.thread)
                    .mapToObj(id -> service.submit(() -> {
                        Thread.currentThread().setName("FT-TrainThread-" + id);
                        trainThread(id);
                        return id;
                    })).collect(Collectors.toSet());
            service.shutdown();
            for (Future<Integer> f : res) {
                try {
                    f.get();
                } catch (InterruptedException e) {
                    System.err.println("Interrupted");
                } catch (ExecutionException e) {
                    res.forEach(_f -> _f.cancel(true));
                    throw e;
                }
            }
        } else {
            trainThread(0);
        }
        model_ = new Model(input_, output_, args_, 0);

        if (args_.verbose > 1) {
            long trainTime = (System.currentTimeMillis() - start_) / 1000;
            out.printf("\nTrain time used: %d sec\n", trainTime);
        }

        saveModel();
        saveVectors();
        if (args_.saveOutput > 0) {
            saveOutput();
        }
    }

    /**
     * The original code:
     * <pre>{@code
     * void FastText::trainThread(int32_t threadId) {
     *  std::ifstream ifs(args_->input);
     *  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);
     *  Model model(input_, output_, args_, threadId);
     *  if (args_->model == model_name::sup) {
     *      model.setTargetCounts(dict_->getCounts(entry_type::label));
     *  } else {
     *      model.setTargetCounts(dict_->getCounts(entry_type::word));
     *  }
     *  const int64_t ntokens = dict_->ntokens();
     *  int64_t localTokenCount = 0;
     *  std::vector<int32_t> line, labels;
     *  while (tokenCount < args_->epoch * ntokens) {
     *      real progress = real(tokenCount) / (args_->epoch * ntokens);
     *      real lr = args_->lr * (1.0 - progress);
     *      if (args_->model == model_name::sup) {
     *          localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
     *          supervised(model, lr, line, labels);
     *      } else if (args_->model == model_name::cbow) {
     *          localTokenCount += dict_->getLine(ifs, line, model.rng);
     *          cbow(model, lr, line);
     *      } else if (args_->model == model_name::sg) {
     *          localTokenCount += dict_->getLine(ifs, line, model.rng);
     *          skipgram(model, lr, line);
     *      }
     *      if (localTokenCount > args_->lrUpdateRate) {
     *          tokenCount += localTokenCount;
     *          localTokenCount = 0;
     *          if (threadId == 0 && args_->verbose > 1) {
     *              printInfo(progress, model.getLoss());
     *          }
     *      }
     *  }
     *  if (threadId == 0 && args_->verbose > 0) {
     *      printInfo(1.0, model.getLoss());
     *      std::cerr << std::endl;
     *  }
     *  ifs.close();
     * }
     * }</pre>
     *
     * @param threadId, int thread identifier
     * @throws IOException if an I/O error occurs
     */
    private void trainThread(int threadId) throws IOException {
        try (FTReader r = args_.createReader()) {
            long skip = threadId * threadFileSize / args_.thread;
            r.skipBytes(skip);
            Model model = new Model(input_, output_, args_, threadId);
            if (args_.model == Args.ModelName.SUP) {
                model.setTargetCounts(dict_.getCounts(EntryType.LABEL));
            } else {
                model.setTargetCounts(dict_.getCounts(EntryType.WORD));
            }

            long ntokens = dict_.ntokens();
            long localTokenCount = 0;
            List<Integer> line = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            while (tokenCount_.longValue() < args_.epoch * ntokens) {
                float progress = tokenCount_.floatValue() / (args_.epoch * ntokens);
                float lr = (float) (args_.lr * (1.0 - progress));
                if (args_.model == Args.ModelName.SUP) {
                    localTokenCount += dict_.getLine(r, line, labels);
                    supervised(model, lr, line, labels);
                } else if (args_.model == Args.ModelName.CBOW) {
                    localTokenCount += dict_.getLine(r, line, model.rng);
                    cbow(model, lr, line);
                } else if (args_.model == ModelName.SG) {
                    localTokenCount += dict_.getLine(r, line, model.rng);
                    skipgram(model, lr, line);
                }
                if (localTokenCount > args_.lrUpdateRate) {
                    tokenCount_.addAndGet(localTokenCount);
                    localTokenCount = 0;
                    if (threadId == 0 && args_.verbose > 1) {
                        printInfo(progress, model.getLoss());
                    }
                }
            }
        }
    }

    public Args getArgs() {
        return args_;
    }

    public Dictionary getDict() {
        return dict_;
    }

    public void setDict(Dictionary dict) {
        this.dict_ = dict;
    }

    public Matrix getInput() {
        return input_;
    }

    public void setInput(Matrix input) {
        this.input_ = input;
    }

    public Matrix getOutput() {
        return output_;
    }

    public void setOutput(Matrix output) {
        this.output_ = output;
    }

    public Model getModel() {
        return model_;
    }

    public void setModel(Model model) {
        this.model_ = model;
    }

}
