package cc.fasttext;

import java.io.*;
import java.lang.ref.Reference;
import java.lang.ref.SoftReference;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.util.FastMath;

import cc.fasttext.Args.ModelName;
import cc.fasttext.Dictionary.EntryType;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
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

    private long threadFileSize;

    public FastText(Args args) {
        this.args_ = args;
    }

    /**
     * Sets output to log.
     *
     * @param out {@link PrintStream} set output to log
     */
    public void setPrintOut(PrintStream out) { // todo: to args or somewhere
        this.out = Objects.requireNonNull(out, "Null out");
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
        Vector res = new Vector(args_.dim);
        List<Integer> ngrams = dict_.getSubwords(word);
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
        Vector res = new Vector(args_.dim);
        if (ModelName.SUP.equals(args_.model)) {
            FTReader in = new FTReader(new ByteArrayInputStream(line.getBytes(args_.getCharset())), args_.getCharset());
            List<Integer> words = new ArrayList<>();
            dict_.getLine(in, words, new ArrayList<>());
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
     * @return
     */
    private Matrix precomputeWordVectors() {
        out.print("Pre-computing word vectors...");
        Matrix res = new Matrix(dict_.nwords(), args_.dim);
        for (int i = 0; i < dict_.nwords(); i++) {
            String word = dict_.getWord(i);
            Vector vec = getWordVector(word);
            float norm = vec.norm();
            if (norm > 0) {
                res.addRow(vec, i, 1.0f / norm);
            }
        }
        out.println(" done.");
        return res;
    }

    private Reference<Matrix> precomputedWordVectors;

    Matrix getPrecomputedWordVectors() {
        Matrix res;
        if (precomputedWordVectors != null && (res = precomputedWordVectors.get()) != null) {
            return res;
        }
        precomputedWordVectors = new SoftReference<>(res = precomputeWordVectors());
        return res;
    }

    private static final double FIND_NN_THRESHOLD = 1e-8;

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
    Multimap<Float, String> findNN(Matrix wordVectors, Vector queryVec, int k, Set<String> banSet) {
        float queryNorm = queryVec.norm();
        if (FastMath.abs(queryNorm) < FIND_NN_THRESHOLD) {
            queryNorm = 1;
        }

        TreeMultimap<Float, String> heap = TreeMultimap.create(Comparator.reverseOrder(), Comparator.reverseOrder());
        Multimap<Float, String> res = TreeMultimap.create(Comparator.reverseOrder(), Comparator.reverseOrder());
        for (int i = 0; i < dict_.nwords(); i++) {
            String word = dict_.getWord(i);
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
        Vector query = new Vector(args_.dim);
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
        dict_.getSubwords(word, ngrams, substrings);
        for (int i = 0; i < ngrams.size(); i++) {
            Vector vec = new Vector(args_.dim);
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
        if (quant_) {
            vec.addRow(qinput_, ind);
        } else {
            vec.addRow(input_, ind);
        }
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
     *      getWordVector(vec, word);
     *      ofs << word << " " << vec << std::endl;
     *  }
     *  ofs.close();
     * }
     *
     * }</pre>
     *
     * @throws IOException
     */
    public void saveVectors() throws IOException {
        if (Utils.isEmpty(args_.output)) { // todo: do we need this validation
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
                Vector vec = getWordVector(word);
                writer.write(word);
                writer.write(" ");
                writer.write(vec.toString());
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

    /**
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
        Objects.requireNonNull(model_, "No model: please load or train it before");

        int nexamples = 0, nlabels = 0;
        double precision = 0.0;
        List<Integer> line = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        FTReader reader = new FTReader(in, args_.getCharset());
        while (!reader.end()) {
            dict_.getLine(reader, line, labels);
            if (labels.isEmpty() || line.isEmpty()) {
                continue;
            }
            TreeMultimap<Float, Integer> modelPredictions = model_.predict(line, k);
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
     * TODO: dont' print. return statistic.
     * Tests and prints to standard output
     *
     * @param in {@link InputStream} to read data
     * @param k  int, positive
     * @throws IOException if wrong i/o
     */
    public void test(InputStream in, int k) throws IOException {
        test(in, this.out, k);
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
        TreeMultimap<Float, Integer> map = model_.predict(words, k, hidden, output);
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
        Objects.requireNonNull(model_, "No model: please load or train it before");
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
        InputStream in = new ByteArrayInputStream(line.getBytes(args_.getCharset().name()));
        FTReader r = new FTReader(in, args_.getCharset());
        return predict(r, k);
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
     * <pre>{@code void FastText::train(std::shared_ptr<Args> args) {
     *  args_ = args;
     *  dict_ = std::make_shared<Dictionary>(args_);
     *  if (args_->input == "-") {
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
     * }}</pre>
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
        startThreads();
        model_ = new Model(input_, output_, args_, 0);
    }

    public void trainAndSave() throws IOException, ExecutionException {
        train();
        saveModel();
        saveVectors();
        if (args_.saveOutput > 0) {
            saveOutput();
        }
    }

    /**
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
     * @throws ExecutionException
     * @throws IOException
     */
    private void startThreads() throws ExecutionException, IOException {
        start_ = System.currentTimeMillis();
        tokenCount_ = new AtomicLong(0);
        if (args_.thread > 1) {
            ExecutorService service = Executors.newFixedThreadPool(args_.thread, r -> {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                t.setDaemon(true);
                return t;
            });
            CompletionService<Void> completionService = new ExecutorCompletionService<>(service);
            Set<Future<Void>> res = IntStream.range(0, args_.thread)
                    .mapToObj(id -> completionService.submit(() -> {
                        Thread.currentThread().setName("FT-TrainThread-" + id);
                        trainThread(id);
                        return null;
                    }))
                    .collect(Collectors.toSet());
            service.shutdown();
            int num = args_.thread;
            try {
                while (num-- > 0) {
                    completionService.take().get();
                }
            } catch (InterruptedException e) {
                out.println("Interrupted!");
                Thread.currentThread().interrupt();
            } finally {
                service.shutdownNow();
            }
        } else {
            trainThread(0);
        }
        if (args_.verbose > 1) {
            long trainTime = (System.currentTimeMillis() - start_) / 1000;
            out.printf("\nTrain time used: %d sec\n", trainTime);
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

    public Matrix getInput() {
        return input_;
    }

    public Matrix getOutput() {
        return output_;
    }

    public Model getModel() {
        return model_;
    }

}
