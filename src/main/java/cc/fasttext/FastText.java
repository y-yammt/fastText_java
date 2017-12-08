package cc.fasttext;

import cc.fasttext.Args.ModelName;
import cc.fasttext.Dictionary.EntryType;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.util.FastMath;
import ru.avicomp.io.FTInputStream;
import ru.avicomp.io.FTOutputStream;
import ru.avicomp.io.FTReader;
import ru.avicomp.io.IOStreams;
import ru.avicomp.io.impl.LocalIOStreams;

import java.io.*;
import java.lang.ref.Reference;
import java.lang.ref.SoftReference;
import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * FastText class, can be used as a lib in other projects.
 * Assuming to be immutable.
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc'>fasttext.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/fasttext.h'>fasttext.h</a>
 *
 * @author Ivan
 */
public strictfp class FastText {
    public static final int FASTTEXT_VERSION = 12;
    public static final int FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

    public static final Factory DEFAULT_FACTORY = new Factory(new LocalIOStreams(), System.out);

    private static final double FIND_NN_THRESHOLD = 1e-8;
    private final Args args;
    private final Dictionary dict;
    private final Model model;
    private final int version;

    private IOStreams fs = DEFAULT_FACTORY.getFileSystem();
    private PrintStream logs = DEFAULT_FACTORY.getLogs();

    private Reference<Matrix> precomputedWordVectors;

    private FastText(Args args, Dictionary dict, Model model, int version) {
        this.args = args;
        this.dict = dict;
        this.model = model;
        this.version = version;
    }

    public static FastText train(Args args, String dataFileURI, String vectorsFileURI) throws IOException, ExecutionException {
        return DEFAULT_FACTORY.train(args, dataFileURI, vectorsFileURI);
    }

    public static FastText load(String modelFileURI) throws IOException, IllegalArgumentException {
        return DEFAULT_FACTORY.load(modelFileURI);
    }

    public Args getArgs() {
        return args;
    }

    public Dictionary getDictionary() {
        return dict;
    }

    public Model getModel() {
        return model;
    }

    public int getVersion() {
        return version;
    }

    public IOStreams getFileSystem() {
        return fs;
    }

    /**
     * Sets file-system.
     *
     * @param fs {@link IOStreams}
     * @return this {@link FastText} object
     */
    public FastText setFileSystem(IOStreams fs) {
        this.fs = Objects.requireNonNull(fs, "Null file system specified.");
        return this;
    }

    /**
     * Sets logs.
     *
     * @param out {@link PrintStream} set output to log
     * @return this {@link FastText} object
     */
    public FastText setLogs(PrintStream out) {
        this.logs = Objects.requireNonNull(out, "Null out");
        return this;
    }

    /**
     * <pre>{@code void FastText::getWordVector(Vector& vec, const std::string& word) const {
     *  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
     *  vec.zero();
     *  for (int i = 0; i < ngrams.size(); i++) {
     *      addInputVector(vec, ngrams[i]);
     *  }
     *  if (ngrams.size() > 0) {
     *      vec.mul(1.0 / ngrams.size());
     *  }
     * }}</pre>
     *
     * @param word String, not null
     * @return {@link Vector}
     */
    public Vector getWordVector(String word) {
        Vector res = new Vector(args.dim());
        List<Integer> ngrams = dict.getSubwords(word);
        for (Integer i : ngrams) {
            addInputVector(res, i);
        }
        if (ngrams.size() > 0) {
            res.mul(1.0f / ngrams.size());
        }
        return res;
    }

    /**
     * <pre>{@code void FastText::getSentenceVector(std::istream& in, fasttext::Vector& svec) {
     *  svec.zero();
     *  if (args_->model == model_name::sup) {
     *      std::vector<int32_t> line, labels;
     *      dict_->getLine(in, line, labels, model_->rng);
     *      for (int32_t i = 0; i < line.size(); i++) {
     *          addInputVector(svec, line[i]);
     *      }
     *      if (!line.empty()) {
     *          svec.mul(1.0 / line.size());
     *      }
     *  } else {
     *      Vector vec(args_->dim);
     *      std::string sentence;
     *      std::getline(in, sentence);
     *      std::istringstream iss(sentence);
     *      std::string word;
     *      int32_t count = 0;
     *      while (iss >> word) {
     *          getWordVector(vec, word);
     *          real norm = vec.norm();
     *          if (norm > 0) {
     *              vec.mul(1.0 / norm);
     *              svec.addVector(vec);
     *              count++;
     *          }
     *      }
     *      if (count > 0) {
     *          svec.mul(1.0 / count);
     *      }
     *  }
     * }}</pre>
     *
     * @return {@link Vector}
     * @throws IOException something wrong while i/o
     */
    public Vector getSentenceVector(String line) throws IOException {
        // add '\n' to the end of line to synchronize behaviour of c++ and java versions
        line += "\n";
        Vector res = new Vector(args.dim());
        if (ModelName.SUP.equals(args.model())) {
            FTReader in = new FTReader(new ByteArrayInputStream(line.getBytes(args.charset())), args.charset());
            List<Integer> words = new ArrayList<>();
            dict.getLine(in, words, new ArrayList<>());
            if (words.isEmpty()) return res;
            for (int w : words) {
                addInputVector(res, w);
            }
            res.mul(1.0f / words.size());
            return res;
        }
        int count = 0;
        for (String word : line.split("\\s+")) {
            Vector vec = getWordVector(word);
            float norm = vec.norm();
            if (norm > 0) {
                vec.mul(1.0f / norm);
                res.addVector(vec);
                count++;
            }
        }
        if (count > 0) {
            res.mul(1.0f / count);
        }
        return res;
    }

    /**
     * <pre>{@code void FastText::precomputeWordVectors(Matrix& wordVectors) {
     *  Vector vec(args_->dim);
     *  wordVectors.zero();
     *  std::cerr << "Pre-computing word vectors...";
     *  for (int32_t i = 0; i < dict_->nwords(); i++) {
     *      std::string word = dict_->getWord(i);
     *      getWordVector(vec, word);
     *      real norm = vec.norm();
     *      if (norm > 0) {
     *          wordVectors.addRow(vec, i, 1.0 / norm);
     *      }
     *  }
     *  std::cerr << " done." << std::endl;
     * }}</pre>
     *
     * @return {@link Matrix}
     */
    private Matrix computeWordVectors() {
        logs.print("Pre-computing word vectors...");
        Matrix res = new Matrix(dict.nwords(), args.dim());
        for (int i = 0; i < dict.nwords(); i++) {
            String word = dict.getWord(i);
            Vector vec = getWordVector(word);
            float norm = vec.norm();
            if (norm > 0) {
                res.addRow(vec, i, 1.0f / norm);
            }
        }
        logs.println(" done.");
        return res;
    }

    Matrix getPrecomputedWordVectors() {
        Matrix res;
        if (precomputedWordVectors != null && (res = precomputedWordVectors.get()) != null) {
            return res;
        }
        precomputedWordVectors = new SoftReference<>(res = computeWordVectors());
        return res;
    }

    /**
     * <pre>{@code
     * void FastText::findNN(const Matrix& wordVectors, const Vector& queryVec, int32_t k, const std::set<std::string>& banSet) {
     *  real queryNorm = queryVec.norm();
     *  if (std::abs(queryNorm) < 1e-8) {
     *      queryNorm = 1;
     *  }
     *  std::priority_queue<std::pair<real, std::string>> heap;
     *  Vector vec(args_->dim);
     *  for (int32_t i = 0; i < dict_->nwords(); i++) {
     *      std::string word = dict_->getWord(i);
     *      real dp = wordVectors.dotRow(queryVec, i);
     *      heap.push(std::make_pair(dp / queryNorm, word));
     *  }
     *  int32_t i = 0;
     *  while (i < k && heap.size() > 0) {
     *      auto it = banSet.find(heap.top().second);
     *      if (it == banSet.end()) {
     *          std::cout << heap.top().second << " " << heap.top().first << std::endl;
     *          i++;
     *      }
     *      heap.pop();
     *  }
     * }
     * }</pre>
     *
     * @param wordVectors
     * @param queryVec
     * @param k
     * @param banSet
     * @return
     */
    private Multimap<Float, String> findNN(Matrix wordVectors, Vector queryVec, int k, Set<String> banSet) {
        float queryNorm = queryVec.norm();
        if (FastMath.abs(queryNorm) < FIND_NN_THRESHOLD) {
            queryNorm = 1;
        }
        TreeMultimap<Float, String> heap = TreeMultimap.create(Comparator.reverseOrder(), Comparator.reverseOrder());
        Multimap<Float, String> res = TreeMultimap.create(Comparator.reverseOrder(), Comparator.reverseOrder());
        for (int i = 0; i < dict.nwords(); i++) {
            String word = dict.getWord(i);
            float dp = wordVectors.dotRow(queryVec, i);
            heap.put(dp / queryNorm, word);
        }
        int i = 0;
        while (i < k && heap.size() > 0) {
            Float key = heap.asMap().firstKey();
            String value = heap.get(key).first();
            if (!banSet.contains(value)) {
                res.put(key, value);
                i++;
            }
            heap.remove(key, value);
        }
        return res;
    }

    /**
     * TODO:
     * <pre>{@code void FastText::nn(int32_t k) {
     *  std::string queryWord;
     *  Vector queryVec(args_->dim);
     *  Matrix wordVectors(dict_->nwords(), args_->dim);
     *  precomputeWordVectors(wordVectors);
     *  std::set<std::string> banSet;
     *  std::cout << "Query word? ";
     *  while (std::cin >> queryWord) {
     *      banSet.clear();
     *      banSet.insert(queryWord);
     *      getWordVector(queryVec, queryWord);
     *      findNN(wordVectors, queryVec, k, banSet);
     *      std::cout << "Query word? ";
     *  }
     * }}</pre>
     *
     * @param k
     * @param queryWord
     * @return
     */
    public Multimap<Float, String> nn(int k, String queryWord) {
        Matrix wordVectors = getPrecomputedWordVectors();
        Set<String> banSet = new HashSet<>();
        banSet.add(queryWord);
        Vector queryVec = getWordVector(queryWord);
        return findNN(wordVectors, queryVec, k, banSet);
    }

    /**
     * <pre>{@code void FastText::analogies(int32_t k) {
     *  std::string word;
     *  Vector buffer(args_->dim), query(args_->dim);
     *  Matrix wordVectors(dict_->nwords(), args_->dim);
     *  precomputeWordVectors(wordVectors);
     *  std::set<std::string> banSet;
     *  std::cout << "Query triplet (A - B + C)? ";
     *  while (true) {
     *      banSet.clear();
     *      query.zero();
     *      std::cin >> word;
     *      banSet.insert(word);
     *      getWordVector(buffer, word);
     *      query.addVector(buffer, 1.0);
     *      std::cin >> word;
     *      banSet.insert(word);
     *      getWordVector(buffer, word);
     *      query.addVector(buffer, -1.0);
     *      std::cin >> word;
     *      banSet.insert(word);
     *      getWordVector(buffer, word);
     *      query.addVector(buffer, 1.0);
     *      findNN(wordVectors, query, k, banSet);
     *      std::cout << "Query triplet (A - B + C)? ";
     * }
     * }}</pre>
     *
     * @param k
     * @param a
     * @param b
     * @param c
     * @return
     */
    public Multimap<Float, String> analogies(int k, String a, String b, String c) {
        Matrix wordVectors = getPrecomputedWordVectors();
        Set<String> banSet = new HashSet<>();
        banSet.add(a);
        Vector query = new Vector(args.dim());
        query.addVector(getWordVector(a), 1.0f);
        banSet.add(b);
        query.addVector(getWordVector(b), -1.0f);
        banSet.add(c);
        query.addVector(getWordVector(c), 1.0f);
        return findNN(wordVectors, query, k, banSet);
    }

    /**
     * <pre>{@code void FastText::ngramVectors(std::string word) {
     *  std::vector<int32_t> ngrams;
     *  std::vector<std::string> substrings;
     *  Vector vec(args_->dim);
     *  dict_->getSubwords(word, ngrams, substrings);
     *  for (int32_t i = 0; i < ngrams.size(); i++) {
     *      vec.zero();
     *      if (ngrams[i] >= 0) {
     *          if (quant_) {
     *              vec.addRow(*qinput_, ngrams[i]);
     *          } else {
     *              vec.addRow(*input_, ngrams[i]);
     *          }
     *      }
     *      std::cout << substrings[i] << " " << vec << std::endl;
     *  }
     * }}</pre>
     *
     * @param out
     * @param word
     */
    void ngramVectors(PrintStream out, String word) {
        List<Integer> ngrams = new ArrayList<>();
        List<String> substrings = new ArrayList<>();
        dict.getSubwords(word, ngrams, substrings);
        for (int i = 0; i < ngrams.size(); i++) {
            Vector vec = new Vector(args.dim());
            if (ngrams.get(i) >= 0) {
                addInputVector(vec, ngrams.get(i));
            }
            out.println(substrings.get(i) + " " + vec);
        }
    }

    /**
     * <pre>{@code void FastText::addInputVector(Vector& vec, int32_t ind) const {
     *  if (quant_) {
     *      vec.addRow(*qinput_, ind);
     *  } else {
     *      vec.addRow(*input_, ind);
     *  }
     * }}</pre>
     *
     * @param vec
     * @param ind
     */
    private void addInputVector(Vector vec, int ind) {
        if (model.isQuant()) {
            vec.addRow(model.qinput(), ind);
        } else {
            vec.addRow(model.input(), ind);
        }
    }

    /**
     * Saves vectors.
     * <p>
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
     *      getWordVector(vec, word);
     *      ofs << word << " " << vec << std::endl;
     *  }
     *  ofs.close();
     * }
     *
     * }</pre>
     *
     * @param file, String file uri path, not null
     * @throws IOException if an I/O error occurs
     */
    public void saveVectors(String file) throws IOException {
        writeVectors("vectors", file, dict.nwords(), dict::getWord, i -> getWordVector(dict.getWord(i)));
    }

    /**
     * Saves output.
     * <p>
     * <pre>{@code void FastText::saveOutput() {
     *  std::ofstream ofs(args_->output + ".output");
     *  if (!ofs.is_open()) {
     *      std::cerr << "Error opening file for saving vectors." << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     *  if (quant_) {
     *      std::cerr << "Option -saveOutput is not supported for quantized models." << std::endl;
     *      return;
     *  }
     *  int32_t n = (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
     *  ofs << n << " " << args_->dim << std::endl;
     *  Vector vec(args_->dim);
     *  for (int32_t i = 0; i < n; i++) {
     *      std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i) : dict_->getWord(i);
     *      vec.zero();
     *      vec.addRow(*output_, i);
     *      ofs << word << " " << vec << std::endl;
     *  }
     *  ofs.close();
     * }
     * }</pre>
     *
     * @param file, String file uri path, not null
     * @throws IOException if an I/O error occurs
     */
    public void saveOutput(String file) throws IOException {
        if (getModel().isQuant()) {
            throw new IllegalStateException("Saving output is not supported for quantized models.");
        }
        writeVectors("output", file,
                ModelName.SUP.equals(args.model()) ? dict.nlabels() : dict.nwords(),
                i -> ModelName.SUP.equals(args.model()) ? dict.getLabel(i) : dict.getWord(i),
                i -> {
                    Vector vec = new Vector(args.dim());
                    vec.addRow(model.output(), i);
                    return vec;
                });
    }

    /**
     * Writes vectors info to the specified file.
     * Auxiliary method.
     *
     * @param name   String, name to log
     * @param file   String, Path (URI) to file
     * @param lines  int first line
     * @param word   function to get String word
     * @param vector function to get {@link Vector vector}
     * @throws IOException in case of io error
     */
    private void writeVectors(String name, String file, int lines, IntFunction<String> word, IntFunction<Vector> vector) throws IOException {
        if (!getFileSystem().canWrite(file)) {
            throw new IOException("Can't write to " + file);
        }
        if (args.verbose() > 1) {
            logs.println("Saving " + name + " to " + file);
        }
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(getFileSystem().createOutput(file), args.charset()))) {
            writer.write(lines + " " + args.dim() + "\n");
            for (int i = 0; i < lines; i++) {
                writer.write(word.apply(i));
                writer.write(" ");
                writer.write(vector.apply(i).toString());
                writer.write("\n");
            }
        }
    }

    /**
     * Writes versions to the model file bin.
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
    private static void signModel(FTOutputStream out) throws IOException {
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
     * @param file
     * @throws IOException
     */
    public void saveModel(String file) throws IOException {
        if (!getFileSystem().canWrite(file)) {
            throw new IOException("Can't write to " + file);
        }
        if (args.verbose() > 1) {
            logs.println("Saving model to " + file);
        }
        try (FTOutputStream out = new FTOutputStream(new BufferedOutputStream(getFileSystem().createOutput(file)))) {
            signModel(out);
            args.save(out);
            dict.save(out);
            boolean quant_ = model.isQuant();
            out.writeBoolean(quant_);
            if (quant_) {
                model.qinput().save(out);
            } else {
                model.input().save(out);
            }
            out.writeBoolean(args.qout());
            if (quant_ && args.qout()) {
                model.qoutput().save(out);
            } else {
                model.output().save(out);
            }
        }
    }

    /**
     * todo: should not accept logs, should return statistic object.
     * Tests.
     * <pre>{@code void FastText::test(std::istream& in, int32_t k) {
     *  int32_t nexamples = 0, nlabels = 0;
     *  double precision = 0.0;
     *  std::vector<int32_t> line, labels;
     *  while (in.peek() != EOF) {
     *      dict_->getLine(in, line, labels, model_->rng);
     *      if (labels.size() > 0 && line.size() > 0) {
     *          std::vector<std::pair<real, int32_t>> modelPredictions;
     *          model_->predict(line, k, modelPredictions);
     *          for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
     *              if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
     *                  precision += 1.0;
     *              }
     *          }
     *          nexamples++;
     *          nlabels += labels.size();
     *      }
     *  }
     *  std::cout << "N" << "\t" << nexamples << std::endl;
     *  std::cout << std::setprecision(3);
     *  std::cout << "P@" << k << "\t" << precision / (k * nexamples) << std::endl;
     *  std::cout << "R@" << k << "\t" << precision / nlabels << std::endl;
     *  std::cerr << "Number of examples: " << nexamples << std::endl;
     * }
     * }</pre>
     *
     * @param in  {@link InputStream} to read data
     * @param out {@link PrintStream} to write data
     * @param k   the number of result labels
     * @throws IOException if something wrong while reading/writing
     */
    public void test(InputStream in, PrintStream out, int k) throws IOException {
        Objects.requireNonNull(in, "Null input");
        Objects.requireNonNull(out, "Null output");
        if (k <= 0) throw new IllegalArgumentException("Negative factor");
        Objects.requireNonNull(model, "No model: please load or train it before");

        int nexamples = 0, nlabels = 0;
        double precision = 0.0;
        List<Integer> line = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        FTReader reader = new FTReader(in, args.charset());
        while (!reader.end()) {
            dict.getLine(reader, line, labels);
            if (labels.isEmpty() || line.isEmpty()) {
                continue;
            }
            TreeMultimap<Float, Integer> modelPredictions = model.predict(line, k);
            precision += modelPredictions.values().stream().filter(labels::contains).count();
            nexamples++;
            nlabels += labels.size();
        }
        out.printf("N\t%d%n", nexamples);
        out.printf(Locale.US, "P@%d: %.3f%n", k, precision / (k * nexamples));
        out.printf(Locale.US, "R@%d: %.3f%n", k, precision / nlabels);
        out.printf(Locale.US, "Number of examples: %d%n", nexamples);
    }

    /**
     * TODO: don't print. return some statistic object.
     * Tests and prints to standard output
     *
     * @param in {@link InputStream} to read data
     * @param k  int, positive
     * @throws IOException if wrong i/o
     */
    public void test(InputStream in, int k) throws IOException {
        test(in, this.logs, k);
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
        dict.getLine(in, words, labels);
        if (words.isEmpty()) {
            return ImmutableListMultimap.of();
        }
        Vector hidden = new Vector(args.dim());
        Vector output = new Vector(dict.nlabels());
        TreeMultimap<Float, Integer> map = model.predict(words, k, hidden, output);
        @SuppressWarnings("ConstantConditions")
        Multimap<String, Float> res = TreeMultimap.create(this::compareLabels, map.keySet().comparator());
        map.forEach((f, i) -> res.put(dict.getLabel(i), f));
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
        if ((dig1 = left.replace(args.label(), "")).matches("\\d+") && (dig2 = right.replace(args.label(), "")).matches("\\d+")) {
            return Integer.compare(Integer.valueOf(dig1), Integer.valueOf(dig2));
        }
        return left.compareTo(right);
    }

    /**
     * todo: must not accept PrintStream: all std-out tasks should be delegated to the {@link Main}
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
        Objects.requireNonNull(model, "No model: please load or train it before");
        FTReader reader = new FTReader(in, args.charset());
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
        predict(in, logs, k, printProb);
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
        InputStream in = new ByteArrayInputStream(line.getBytes(args.charset().name()));
        FTReader r = new FTReader(in, args.charset());
        return predict(r, k);
    }

    /**
     * <pre>{@code std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const {
     *  Vector norms(input_->m_);
     *  input_->l2NormRow(norms);
     *  std::vector<int32_t> idx(input_->m_, 0);
     *  std::iota(idx.begin(), idx.end(), 0);
     *  auto eosid = dict_->getId(Dictionary::EOS);
     *  std::sort(idx.begin(), idx.end(), [&norms, eosid] (size_t i1, size_t i2) {
     *      return eosid ==i1 || (eosid != i2 && norms[i1] > norms[i2]);
     *  });
     *  idx.erase(idx.begin() + cutoff, idx.end());
     *  return idx;
     * }}</pre>
     *
     * @param cutoff
     * @return
     */
    private List<Integer> selectEmbeddings(int cutoff) {
        Vector norms = model.input().l2NormRow();
        List<Integer> idx = IntStream.iterate(0, i -> ++i).limit(model.input().getM()).boxed().collect(Collectors.toList());
        int eosId = dict.getId(Dictionary.EOS);
        idx.sort((i1, i2) -> {
            boolean res = eosId == i1 || (eosId != i2 && norms.get(i1) > norms.get(i2));
            return res ? -1 : 1;
        });
        return idx.subList(0, cutoff);
    }

    /**
     * Creates a quantized model from existing one.
     * Note 1: unlike original c++ method, this one does not change state of current FastText object, so take care about memory!
     * Note 2: Only for supervised models.
     * <p>
     * <pre>{@code void FastText::quantize(std::shared_ptr<Args> qargs) {
     *  if (args_->model != model_name::sup) {
     *      throw std::invalid_argument("For now we only support quantization of supervised models");
     *  }
     *  args_->input = qargs->input;
     *  args_->qout = qargs->qout;
     *  args_->output = qargs->output;
     *  if (qargs->cutoff > 0 && qargs->cutoff < input_->m_) {
     *      auto idx = selectEmbeddings(qargs->cutoff);
     *      dict_->prune(idx);
     *      std::shared_ptr<Matrix> ninput = std::make_shared<Matrix>(idx.size(), args_->dim);
     *      for (auto i = 0; i < idx.size(); i++) {
     *          for (auto j = 0; j < args_->dim; j++) {
     *              ninput->at(i, j) = input_->at(idx[i], j);
     *          }
     *      }
     *      input_ = ninput;
     *      if (qargs->retrain) {
     *          args_->epoch = qargs->epoch;
     *          args_->lr = qargs->lr;
     *          args_->thread = qargs->thread;
     *          args_->verbose = qargs->verbose;
     *          startThreads();
     *      }
     *  }
     *  qinput_ = std::make_shared<QMatrix>(*input_, qargs->dsub, qargs->qnorm);
     *  if (args_->qout) {
     *      qoutput_ = std::make_shared<QMatrix>(*output_, 2, qargs->qnorm);
     *  }
     *  quant_ = true;
     *  model_ = std::make_shared<Model>(input_, output_, args_, 0);
     *  model_->quant_ = quant_;
     *  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);
     * }}</pre>
     *
     * @param other
     * @param dataFileURIToRetrain String, the file uri with data to perform retrain, nullable
     * @return new {@link FastText fasttext model} instance.
     * @throws IOException              if an I/O error occurs while retraining.
     * @throws ExecutionException       if any error occurs while retraining
     * @throws IllegalStateException    in case model is already quantized.
     * @throws IllegalArgumentException if some args are wrong.
     */
    public FastText quantize(Args other, String dataFileURIToRetrain) throws IOException, ExecutionException, IllegalStateException, IllegalArgumentException {
        if (model.isQuant()) {
            throw new IllegalStateException("Already quantized.");
        }
        if (!ModelName.SUP.equals(args.model())) {
            throw new IllegalArgumentException("For now we only support quantization of supervised models");
        }
        Args qargs = new Args.Builder()
                .copy(this.args)
                .setQOut(other.qout())
                .setCutOff(other.cutoff())
                .setQNorm(other.qnorm())
                .setDSub(other.dsub())
                .build();
        Dictionary qdict = this.dict.copy();
        Matrix input;
        Matrix output = model.output().copy();
        if (qargs.cutoff() > 0 && qargs.cutoff() < model.input().getM()) {
            List<Integer> idx = qdict.prune(selectEmbeddings(qargs.cutoff()));
            input = new Matrix(idx.size(), qargs.dim());
            for (int i = 0; i < idx.size(); i++) {
                for (int j = 0; j < qargs.dim(); j++) {
                    input.put(i, j, model.input().at(idx.get(i), j));
                }
            }
            if (!StringUtils.isEmpty(dataFileURIToRetrain)) {
                qargs = new Args.Builder()
                        .copy(qargs)
                        .setEpoch(other.epoch())
                        .setLR(other.lr())
                        .setThread(other.thread())
                        .setVerbose(other.verbose())
                        .build();
                Model model = new Factory(fs, logs).newTrainer(qargs, dataFileURIToRetrain, qdict, input, output).train();
                input = model.input();
                output = model.output();
            }
        } else {
            input = model.input().copy();
        }

        QMatrix qinput = new QMatrix(input, args.randomFactory(), qargs.dsub(), qargs.qnorm());
        QMatrix qoutput;
        if (qargs.qout()) {
            qoutput = new QMatrix(output, args.randomFactory(), 2, qargs.qnorm());
        } else {
            qoutput = new QMatrix();
        }
        Model model = new Model(input, output, qargs, 0).setQuantizePointer(qinput, qoutput);
        return new FastText(qargs, qdict, model, FASTTEXT_VERSION);
    }

    /**
     * A factory to produce new {@link FastText} interface.
     * @see IOStreams
     * <p>
     * Created by @szuev on 07.12.2017.
     */
    public static class Factory {
        // todo: buff size is experimental
        private static final int BUFF_SIZE = 100 * 1024;

        private final IOStreams fs;
        private final PrintStream logs;

        public Factory(IOStreams factory, PrintStream logs) {
            this.fs = Objects.requireNonNull(factory, "Null io-factory.");
            this.logs = Objects.requireNonNull(logs, "Null logs");
        }

        public Factory setFileSystem(IOStreams fs) {
            return new Factory(fs, this.logs);
        }

        public Factory setLogs(PrintStream logs) {
            return new Factory(this.fs, logs);
        }

        public IOStreams getFileSystem() {
            return fs;
        }

        public PrintStream getLogs() {
            return logs;
        }

        /**
         * Loads model by file-reference (URI) using {@link IOStreams file-system}.
         * <p>
         * <pre>{@code void FastText::loadModel(const std::string& filename) {
         *  std::ifstream ifs(filename, std::ifstream::binary);
         *  if (!ifs.is_open()) {
         *      throw std::invalid_argument(filename + " cannot be opened for loading!");
         *  }
         *  if (!checkModel(ifs)) {
         *      throw std::invalid_argument(filename + " has wrong file format!");
         *  }
         *  loadModel(ifs);
         *  ifs.close();
         * }}</pre>
         *
         * @param uri String, path to file, not null
         * @return new {@link FastText model} instance
         * @throws IOException              if something is wrong while read file
         * @throws IllegalArgumentException if file is wrong
         */
        public FastText load(String uri) throws IOException, IllegalArgumentException {
            if (!fs.canRead(Objects.requireNonNull(uri, "Null file ref specified."))) {
                throw new IOException("Model file cannot be opened for loading: <" + uri + ">");
            }
            try (InputStream in = fs.openInput(uri)) {
                logs.print("Load model " + uri + " ...");
                FastText res = load(in).setFileSystem(fs).setLogs(logs);
                logs.println(" done.");
                return res;
            } catch (Exception e) {
                logs.println(" error.");
                throw e;
            }
        }

        /**
         * Loads model from any InputStream.
         * <p>
         * Original methods:
         * <pre>{@code void FastText::loadModel(std::istream& in) {
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
         *      std::cerr << "Invalid model file.\n"  << "Please download the updated model from www.fasttext.cc.\n"
         *          << "See issue #332 on Github for more information.\n";
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
         * @param in {@link InputStream}
         * @return new {@link FastText model} instance
         * @throws IOException              if something is wrong while read file
         * @throws IllegalArgumentException if file is wrong
         */
        public FastText load(InputStream in) throws IOException, IllegalArgumentException {
            FTInputStream inputStream = new FTInputStream(new BufferedInputStream(in));
            int magic = inputStream.readInt();
            if (FASTTEXT_FILEFORMAT_MAGIC_INT32 != magic) {
                throw new IllegalArgumentException("Model file has wrong format!");
            }
            int version = inputStream.readInt();
            if (version > FASTTEXT_VERSION) {
                throw new IllegalArgumentException("Model file has wrong format!");
            }
            Args args = Args.load(inputStream);
            if (version == 11 && args.model() == ModelName.SUP) {
                // backward compatibility: old supervised models do not use char ngrams.
                args = new Args.Builder().copy(args).setMaxN(0).build();
            }
            Dictionary dict = new Dictionary(args);
            dict.load(inputStream);
            boolean quantInput = inputStream.readBoolean();
            Matrix input = new Matrix();
            QMatrix qinput = new QMatrix();
            boolean quant;
            if (quantInput) {
                qinput.load(inputStream);
                quant = true;
            } else {
                input.load(inputStream);
                quant = false;
            }

            if (!quantInput && dict.isPruned()) {
                throw new IllegalArgumentException("Invalid model file.\nPlease download the updated model from " +
                        "www.fasttext.cc.\nSee issue #332 on Github for more information.\n");
            }
            // todo warn: after following line different args in the dictionary and this object:
            args = new Args.Builder().copy(args).setQOut(inputStream.readBoolean()).build();
            Matrix output = new Matrix();
            QMatrix qoutput = new QMatrix();
            if (quant && args.qout()) {
                qoutput.load(inputStream);
            } else {
                output.load(inputStream);
            }

            Model model = new Model(input, output, args, 0).setQuantizePointer(qinput, qoutput);
            if (ModelName.SUP.equals(args.model())) {
                model.setTargetCounts(dict.getCounts(EntryType.LABEL));
            } else {
                model.setTargetCounts(dict.getCounts(EntryType.WORD));
            }
            return new FastText(args, dict, model, version);
        }

        /**
         * Loads matrix from file.
         * <p>
         * <pre>{@code void FastText::loadVectors(std::string filename) {
         *  std::ifstream in(filename);
         *  std::vector<std::string> words;
         *  std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
         *  int64_t n, dim;
         *  if (!in.is_open()) {
         *      std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
         *      exit(EXIT_FAILURE);
         *  }
         *  in >> n >> dim;
         *  if (dim != args_->dim) {
         *      std::cerr << "Dimension of pretrained vectors does not match -dim option" << std::endl;
         *      exit(EXIT_FAILURE);
         *  }
         *  mat = std::make_shared<Matrix>(n, dim);
         *  for (size_t i = 0; i < n; i++) {
         *      std::string word;
         *      in >> word;
         *      words.push_back(word);
         *      dict_->add(word);
         *      for (size_t j = 0; j < dim; j++) {
         *          in >> mat->data_[i * dim + j];
         *      }
         *  }
         *  in.close();
         *  dict_->threshold(1, 0);
         *  input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
         *  input_->uniform(1.0 / args_->dim);
         *  for (size_t i = 0; i < n; i++) {
         *      int32_t idx = dict_->getId(words[i]);
         *      if (idx < 0 || idx >= dict_->nwords()) continue;
         *      for (size_t j = 0; j < dim; j++) {
         *          input_->data_[idx * dim + j] = mat->data_[i * dim + j];
         *      }
         *  }
         * }}</pre>
         *
         * @param args       {@link Args} args object
         * @param dictionary {@link Dictionary} object
         * @param file       String, uri to file
         * @return {@link Matrix}, the input matrix to construct new model
         * @throws IOException              if some error during reading file occurs
         * @throws IllegalArgumentException if some input arguments are wrong
         * @see FastText#saveVectors(String)
         */
        private Matrix loadInput(Args args, Dictionary dictionary, String file) throws IOException, IllegalArgumentException {
            if (!fs.canRead(file)) {
                throw new IllegalArgumentException("Pre-trained vectors file cannot be opened!");
            }
            Matrix mat;
            List<String> words;
            int n, dim;
            try (BufferedReader in = new BufferedReader(new InputStreamReader(fs.openInput(file), args.charset()))) {
                String first = in.readLine();
                if (!first.matches("\\d+\\s+\\d+")) {
                    throw new IllegalArgumentException("Wrong pre-trained vectors file: first line should contain 'n dim' pair");
                }
                n = Integer.parseInt(first.split("\\s+")[0]);
                dim = Integer.parseInt(first.split("\\s+")[1]);
                if (dim != args.dim()) {
                    throw new IllegalArgumentException("Dimension of pretrained vectors does not match -dim option: found " + dim + ", expected " + args.dim());
                }
                mat = new Matrix(n, dim);
                words = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    String line = in.readLine();
                    String word;
                    String[] array;
                    if (StringUtils.isEmpty(line) || (array = line.split("\\s+")).length == 0 || StringUtils.isEmpty(word = array[0])) {
                        throw new IllegalArgumentException("Wrong line: " + line);
                    }
                    List<Float> numbers = Arrays.stream(array).skip(1).limit(dim + 1).map(Float::parseFloat).collect(Collectors.toList());
                    if (numbers.size() < dim)
                        throw new IllegalArgumentException("Wrong numbers in the line: " + numbers.size() + ". Expected " + dim);
                    words.add(word);
                    for (int j = 0; j < numbers.size(); j++) {
                        mat.set(i, j, numbers.get(j));
                    }
                }
            }
            dictionary.threshold(1, 0);
            Matrix res = new Matrix(dictionary.nwords() + args.bucket(), args.dim());
            res.uniform(args.randomFactory().apply(1), 1.0f / args.dim());
            for (int i = 0; i < n; i++) {
                int idx = dictionary.getId(words.get(i));
                if (idx < 0 || idx >= dictionary.nwords())
                    continue;
                for (int j = 0; j < dim; j++) {
                    res.set(idx, j, mat.get(i, j));
                }
            }
            return res;
        }

        /**
         * Reads dictionary from file.
         *
         * @param args {@link Args} settings to construct new dictionary
         * @param file String, file path, not null
         * @return {@link Dictionary}
         * @throws IOException if an I/O error occurs
         */
        private Dictionary readDictionary(Args args, String file) throws IOException {
            Dictionary res = new Dictionary(args);
            try (FTReader reader = createReader(args, file)) {
                res.readFromFile(reader, logs);
            }
            return res;
        }

        private FTReader createReader(Args args, String fileName) throws IOException {
            return new FTReader(fs.openScrollable(fileName), args.charset(), BUFF_SIZE);
        }

        private Matrix createInput(Args args, Dictionary dictionary) {
            Matrix res = new Matrix(dictionary.nwords() + args.bucket(), args.dim());
            res.uniform(args.randomFactory().apply(1), 1.0f / args.dim());
            return res;
        }

        private Matrix createOutput(Args args, Dictionary dictionary) {
            if (ModelName.SUP.equals(args.model())) {
                return new Matrix(dictionary.nlabels(), args.dim());
            }
            return new Matrix(dictionary.nwords(), args.dim());
        }

        private Trainer newTrainer(Args args, String file, String vectors) throws IOException {
            if (!fs.canRead(Objects.requireNonNull(file, "Null data file specified"))) {
                throw new IllegalArgumentException("Input file cannot be opened: " + file);
            }
            long size = fs.size(file);
            Dictionary dic = readDictionary(args, file);
            Matrix in = vectors == null ? createInput(args, dic) : loadInput(args, dic, vectors);
            Matrix out = createOutput(args, dic);
            return new Trainer(args, file, size, dic, in, out);
        }

        private Trainer newTrainer(Args args, String file, Dictionary dictionary, Matrix input, Matrix output) throws IOException {
            if (!fs.canRead(Objects.requireNonNull(file, "Null data file specified"))) {
                throw new IllegalArgumentException("Input file cannot be opened: " + file);
            }
            long size = fs.size(file);
            return new Trainer(args, file, size, dictionary, input, output);
        }

        public FastText train(Args args, String file, String vectors) throws IOException, ExecutionException {
            Trainer trainer = newTrainer(args, file, vectors);
            Model model = trainer.train();
            return new FastText(args, trainer.dictionary, model, FASTTEXT_VERSION).setFileSystem(fs).setLogs(logs);
        }

        private void printInfo(Args args, Instant start, Instant end, long tokenCount_, float progress, float loss) {
            Duration d = Duration.between(start, end);
            float t = d.get(ChronoUnit.SECONDS) + d.get(ChronoUnit.NANOS) / 1_000_000_000f;
            float wst = tokenCount_ / t;
            float lr = (float) (args.lr() * (1 - progress));
            int eta = (int) (t / progress * (1 - progress) / args.thread());
            int etaH = eta / 3600;
            int etaM = (eta - etaH * 3600) / 60;
            logs.printf("\rProgress: %.1f%% words/sec/thread: %.0f lr: %.6f loss: %.6f eta: %d h %d m", 100 * progress, wst, lr, loss, etaH, etaM);
            if (progress == 1) {
                logs.println();
            }
        }

        /**
         * Auxiliary class to perform model training.
         */
        private class Trainer {
            private final String file;
            private final long size;

            private final Args args;
            private final Dictionary dictionary;

            private final Matrix input;
            private final Matrix output;

            private Instant start;
            private AtomicLong tokenCount;

            private Trainer(Args args, String file, long size, Dictionary dictionary, Matrix input, Matrix output) {
                this.args = Objects.requireNonNull(args, "Null args");
                this.file = Objects.requireNonNull(file, "Null file");
                this.size = size;
                this.dictionary = Objects.requireNonNull(dictionary, "Null dictionary");
                this.input = Objects.requireNonNull(input, "Null input matrix");
                this.output = Objects.requireNonNull(output, "Null output matrix");
            }

            private FTReader createReader() throws IOException {
                return Factory.this.createReader(args, file);
            }

            /**
             * <pre>{@code void FastText::train(std::shared_ptr<Args> args) {
             *  args_ = args;
             *  dict_ = std::make_shared<Dictionary>(args_);
             *  if (args_->input == "-") { // manage expectations
             *      std::cerr << "Cannot use stdin for training!" << std::endl;
             *      exit(EXIT_FAILURE);
             *  }
             *  std::ifstream ifs(args_->input);
             *  if (!ifs.is_open()) {
             *      std::cerr << "Input file cannot be opened!" << std::endl;
             *      exit(EXIT_FAILURE);
             *  }
             *  dict_->readFromFile(ifs);
             *  ifs.close();
             *  if (args_->pretrainedVectors.size() != 0) {
             *      loadVectors(args_->pretrainedVectors);
             *  } else {
             *      input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
             *      input_->uniform(1.0 / args_->dim);
             *  }
             *  if (args_->model == model_name::sup) {
             *      output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
             *  } else {
             *      output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
             *  }
             *  output_->zero();
             *  startThreads();
             *  model_ = std::make_shared<Model>(input_, output_, args_, 0);
             *  if (args_->model == model_name::sup) {
             *      model_->setTargetCounts(dict_->getCounts(entry_type::label));
             *  } else {
             *      model_->setTargetCounts(dict_->getCounts(entry_type::word));
             *  }
             * }}</pre>
             *
             * @return {@link Model} new instance
             * @throws IOException              if something wrong while reading file
             * @throws ExecutionException       if something wrong while training in multithreading
             * @throws IllegalArgumentException in case wrong file refs.
             */
            public Model train() throws IOException, ExecutionException, IllegalArgumentException {
                startThreads();
                return new Model(input, output, args, 0);
            }

            /**
             * Runs training treads and waits for finishing.
             * <p>
             * <pre>{@code void FastText::startThreads() {
             *  start = clock();
             *  tokenCount = 0;
             *  if (args_->thread > 1) {
             *      std::vector<std::thread> threads;
             *      for (int32_t i = 0; i < args_->thread; i++) {
             *          threads.push_back(std::thread([=]() { trainThread(i); }));
             *      }
             *      for (auto it = threads.begin(); it != threads.end(); ++it) {
             *          it->join();
             *      }
             *  } else {
             *      trainThread(0);
             *  }
             * }}</pre>
             *
             * @throws ExecutionException if any error occurs in any sub-treads
             * @throws IOException        if an I/O error occurs
             */
            private void startThreads() throws ExecutionException, IOException {
                this.start = Instant.now();
                this.tokenCount = new AtomicLong(0);
                BiConsumer<Float, Float> zeroPrinter = (progress, loss) -> {
                };
                BiConsumer<Float, Float> printer = args.verbose() > 1 ? this::printInfo : zeroPrinter;
                if (args.thread() <= 1) {
                    trainThread(0, printer);
                    return;
                }
                ExecutorService service = Executors.newFixedThreadPool(args.thread(), r -> {
                    Thread t = Executors.defaultThreadFactory().newThread(r);
                    t.setDaemon(true);
                    return t;
                });
                CompletionService<Void> completionService = new ExecutorCompletionService<>(service);
                IntStream.range(0, args.thread()).forEach(id ->
                        completionService.submit(() -> {
                            Thread.currentThread().setName("FT-TrainThread-" + id);
                            trainThread(id, id == 0 ? printer : zeroPrinter);
                            return null;
                        }));
                service.shutdown();
                int num = args.thread();
                try {
                    while (num-- > 0) {
                        completionService.take().get();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    service.shutdownNow();
                }
            }

            /**
             * <pre>{@code void FastText::trainThread(int32_t threadId) {
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
             * }}</pre>
             *
             * @param threadId the id of thread, used as random seed
             * @param printer  auxiliary object to print logs
             * @throws IOException if an I/O error occurs
             */
            private void trainThread(int threadId, BiConsumer<Float, Float> printer) throws IOException {
                Model model = new Model(input, output, args, threadId);
                try (FTReader reader = createReader()) {
                    long skip = threadId * size / args.thread();
                    reader.skipBytes(skip);
                    if (ModelName.SUP.equals(args.model())) {
                        model.setTargetCounts(dictionary.getCounts(EntryType.LABEL));
                    } else {
                        model.setTargetCounts(dictionary.getCounts(EntryType.WORD));
                    }
                    long ntokens = dictionary.ntokens();
                    long localTokenCount = 0;
                    List<Integer> line = new ArrayList<>();
                    List<Integer> labels = new ArrayList<>();
                    while (tokenCount.longValue() < args.epoch() * ntokens) {
                        float progress = tokenCount.floatValue() / (args.epoch() * ntokens);
                        float lr = (float) (args.lr() * (1.0 - progress));
                        if (ModelName.SUP.equals(args.model())) {
                            localTokenCount += dictionary.getLine(reader, line, labels);
                            supervised(model, lr, line, labels);
                        } else if (ModelName.CBOW.equals(args.model())) {
                            localTokenCount += dictionary.getLine(reader, line, model.rng);
                            cbow(model, lr, line);
                        } else if (ModelName.SG.equals(args.model())) {
                            localTokenCount += dictionary.getLine(reader, line, model.rng);
                            skipgram(model, lr, line);
                        }
                        if (localTokenCount > args.lrUpdateRate()) {
                            tokenCount.addAndGet(localTokenCount);
                            localTokenCount = 0;
                            printer.accept(progress, model.getLoss());
                        }
                    }
                }
                printer.accept(1f, model.getLoss());
            }

            /**
             * <pre>{@code
             * void FastText::supervised(Model& model, real lr, const std::vector<int32_t>& line, const std::vector<int32_t>& labels) {
             *  if (labels.size() == 0 || line.size() == 0) return;
             *  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
             *  int32_t i = uniform(model.rng);
             *  model.update(line, labels[i], lr);
             * }}</pre>
             *
             * @param model  {@link Model}
             * @param lr     float
             * @param line   List of ints
             * @param labels List of ints
             */
            private void supervised(Model model, float lr, List<Integer> line, List<Integer> labels) {
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
             * }}</pre>
             *
             * @param model {@link Model}
             * @param lr    float
             * @param line  List of ints
             */
            private void cbow(Model model, float lr, List<Integer> line) {
                UniformIntegerDistribution uniform = new UniformIntegerDistribution(model.rng, 1, args.ws());
                for (int w = 0; w < line.size(); w++) {
                    List<Integer> bow = new ArrayList<>();
                    int boundary = uniform.sample();
                    for (int c = -boundary; c <= boundary; c++) {
                        int wc;
                        if (c != 0 && (wc = w + c) >= 0 && wc < line.size()) {
                            List<Integer> ngrams = dictionary.getSubwords(line.get(wc));
                            bow.addAll(ngrams);
                        }
                    }
                    model.update(bow, line.get(w), lr);
                }
            }

            /**
             * <pre>{@code void FastText::skipgram(Model& model, real lr, const std::vector<int32_t>& line) {
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
             * @param model {@link Model}
             * @param lr    float
             * @param line  List of ints
             */
            private void skipgram(Model model, float lr, List<Integer> line) {
                UniformIntegerDistribution uniform = new UniformIntegerDistribution(model.rng, 1, args.ws());
                for (int w = 0; w < line.size(); w++) {
                    int boundary = uniform.sample();
                    List<Integer> ngrams = dictionary.getSubwords(line.get(w));
                    for (int c = -boundary; c <= boundary; c++) {
                        int wc;
                        if (c != 0 && (wc = w + c) >= 0 && wc < line.size()) {
                            model.update(ngrams, line.get(wc), lr);
                        }
                    }
                }
            }

            /**
             * Prints debug train info to console or somewhere else.
             * <p>
             * <pre>{@code void FastText::printInfo(real progress, real loss) {
             *  real t = real(clock() - start) / CLOCKS_PER_SEC;
             *  real wst = real(tokenCount) / t;
             *  real lr = args_->lr * (1.0 - progress);
             *  int eta = int(t / progress * (1 - progress) / args_->thread);
             *  int etah = eta / 3600;
             *  int etam = (eta - etah * 3600) / 60;
             *  std::cerr << std::fixed;
             *  std::cerr << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
             *  std::cerr << "  words/sec/thread: " << std::setprecision(0) << wst;
             *  std::cerr << "  lr: " << std::setprecision(6) << lr;
             *  std::cerr << "  loss: " << std::setprecision(6) << loss;
             *  std::cerr << "  eta: " << etah << "h" << etam << "m ";
             *  std::cerr << std::flush;
             * }}</pre>
             *
             * @param progress float
             * @param loss     float
             */
            private void printInfo(float progress, float loss) {
                Factory.this.printInfo(args, start, Instant.now(), tokenCount.get(), progress, loss);
            }
        }
    }

}
