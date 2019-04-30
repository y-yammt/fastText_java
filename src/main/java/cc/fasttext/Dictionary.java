package cc.fasttext;

import cc.fasttext.io.*;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.primitives.Floats;
import com.google.common.primitives.UnsignedLong;
import org.apache.commons.lang.Validate;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.Charset;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.ToLongFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The dictionary.
 * @see <a href='https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc'>dictionary.cc</a>
 * @see <a href='https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h'>dictionary.h</a>
 */
public class Dictionary {

    public static final String BOW = "<";
    public static final String EOW = ">";
    public static final String EOS = BOW + "/s" + EOW;
    public static final String DELIMITERS = "\n\r\t \u000b\f\0";

    public static final int MAX_VOCAB_SIZE = 30_000_000;
    public static final int MAX_LINE_SIZE = 1024;
    private static final Integer WORD_ID_DEFAULT = -1;
    private static final Integer PRUNE_IDX_SIZE_DEFAULT = -1;

    private static final UnsignedLong ADD_WORDS_NGRAMS_FACTOR_UNSIGNED_LONG = UnsignedLong.valueOf(116_049_371L);

    private static final long READ_LOG_STEP = 1_000_000;

    private static final int PARALLEL_SIZE_THRESHOLD = Integer.parseInt(System.getProperty("parallel.dictionary.threshold",
            String.valueOf(FastText.PARALLEL_THRESHOLD_FACTOR * 100)));

    private static final Comparator<Entry> ENTRY_COMPARATOR = Comparator.comparing((Function<Entry, EntryType>) t -> t.type)
            .thenComparing(Comparator.comparingLong((ToLongFunction<Entry>) value -> value.count).reversed());

    private List<Entry> words = new ArrayList<>(MAX_VOCAB_SIZE);
    private List<Float> pdiscard;
    private Map<Long, Integer> word2int = new HashMap<>(MAX_VOCAB_SIZE);
    private int size;
    private int nwords;
    private int nlabels;
    private long ntokens;
    private long pruneIdxSize = PRUNE_IDX_SIZE_DEFAULT;
    private Map<Integer, Integer> pruneIdx = new HashMap<>();
    private final Charset charset;

    // args:
    private final UnsignedLong bucket;
    private final int maxn;
    private final int minn;
    private final int wordNgrams;
    private final Args.ModelName model;
    private final String label;
    private final double t;

    Dictionary(Args args, Charset charset) {
        this(args.model(), args.label(), args.samplingThreshold(), UnsignedLong.valueOf(args.bucket()), args.maxn(), args.minn(), args.wordNgrams(), charset);
    }

    private Dictionary(Args.ModelName model, String label, double samplingThreshold, UnsignedLong bucket, int maxn, int minn, int wordNgrams, Charset charset) {
        this.model = model;
        this.label = label;
        this.bucket = bucket;
        this.t = samplingThreshold;
        this.maxn = maxn;
        this.minn = minn;
        this.wordNgrams = wordNgrams;
        this.charset = charset;
    }

    public Charset charset() {
        return charset;
    }

    long find(String w) {
        return find(w, hash(w));
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code int32_t Dictionary::find(const std::string& w, uint32_t h) const {
     *  int32_t word2intsize = word2int_.size();
     *  int32_t id = h % word2intsize;
     *  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
     *    id = (id + 1) % word2intsize;
     *  }
     *  return id;
     * }}</pre>
     * <pre>{@code int32_t Dictionary::find(const std::string& w, uint32_t h) const {
     *  int32_t id = h % MAX_VOCAB_SIZE;
     *  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
     *      id = (id + 1) % MAX_VOCAB_SIZE;
     *  }
     *  return id;
     * }}</pre>
     *
     * @param w String
     * @param h long (uint32_t)
     * @return int (int32_t)
     */
    private long find(String w, long h) {
        long id = h % MAX_VOCAB_SIZE;
        while (!Objects.equals(word2int.getOrDefault(id, WORD_ID_DEFAULT), WORD_ID_DEFAULT) &&
                !Objects.equals(words.get(word2int.get(id)).word, w)) {
            id = (id + 1) % MAX_VOCAB_SIZE;
        }
        return id;
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code void Dictionary::add(const std::string& w) {
     *  int32_t h = find(w);
     *  ntokens_++;
     *  if (word2int_[h] == -1) {
     *    entry e;
     *    e.word = w;
     *    e.count = 1;
     *    e.type = getType(w);
     *    words_.push_back(e);
     *    word2int_[h] = size_++;
     *  } else {
     *    words_[word2int_[h]].count++;
     *  }
     * }}</pre>
     * <pre>{@code void Dictionary::add(const std::string& w) {
     *  int32_t h = find(w);
     *  ntokens_++;
     *  if (word2int_[h] == -1) {
     *      entry e;
     *      e.word = w;
     *      e.count = 1;
     *      e.type = getType(w);
     *      words_.push_back(e);
     *      word2int_[h] = size_++;
     *  } else {
     *      words_[word2int_[h]].count++;
     *  }
     * }}</pre>
     *
     * @param w String
     */
    void add(String w) {
        long h = find(w);
        ntokens++;
        if (Objects.equals(word2int.getOrDefault(h, WORD_ID_DEFAULT), WORD_ID_DEFAULT)) {
            Entry e = new Entry(w, 1, getType(w));
            words.add(e);
            word2int.put(h, size++);
        } else {
            words.get(word2int.get(h)).count++;
        }
    }

    public int nwords() {
        return nwords;
    }

    public int nlabels() {
        return nlabels;
    }

    public long ntokens() {
        return ntokens;
    }

    public List<Entry> getWords() {
        return words;
    }

    public int size() {
        return size;
    }

    List<Float> pdiscard() {
        return pdiscard;
    }

    Map<Long, Integer> getWord2int() {
        return word2int;
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code int32_t Dictionary::getId(const std::string& w) const {
     *  int32_t h = find(w);
     *  return word2int_[h];
     * }}</pre>
     * <pre>{@code int32_t Dictionary::getId(const std::string& w) const {
     *  int32_t h = find(w);
     *  return word2int_[h];
     * }}</pre>
     *
     * @param w String
     * @return int32_t
     */
    public int getId(String w) {
        return word2int.getOrDefault(find(w), WORD_ID_DEFAULT);
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
     *  int32_t id = find(w, h);
     *  return word2int_[id];
     * }}</pre>
     * <pre>{@code int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
     *  int32_t id = find(w, h);
     *  return word2int_[id];
     * }}</pre>
     *
     * @param w String
     * @param h long (uint32_t)
     * @return int (default: -1)
     */
    private int getId(String w, long h) {
        return word2int.getOrDefault(find(w, h), WORD_ID_DEFAULT);
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code entry_type Dictionary::getType(const std::string& w) const {
     *  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
     * }}</pre>
     * <pre>{@code entry_type Dictionary::getType(const std::string& w) const {
     *  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
     * }}</pre>
     *
     * @param w String
     * @return {@link EntryType}
     */
    private EntryType getType(String w) {
        return w.startsWith(label) ? EntryType.LABEL : EntryType.WORD;
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code entry_type Dictionary::getType(int32_t id) const {
     *  assert(id >= 0);
     *  assert(id < size_);
     *  return words_[id].type;
     * }}</pre>
     * <pre>{@code entry_type Dictionary::getType(int32_t id) const {
     *  assert(id >= 0);
     *  assert(id < size_);
     *  return words_[id].type;
     * }}</pre>
     *
     * @param id int32_r
     * @return {@link EntryType}
     */
    private EntryType getType(int id) {
        Validate.isTrue(id >= 0);
        Validate.isTrue(id < size);
        return words.get(id).type;
    }

    public String getWord(int id) {
        Validate.isTrue(id >= 0);
        Validate.isTrue(id < size);
        return words.get(id).word;
    }

    /**
     * String FNV-1a Hash
     * Test data:
     * <pre>
     * 2166136261      ''
     * 3826002220      'a'
     * 805092869       'Test'
     * 386908734       'This is some test sentence.'
     * 1487114043      '这是一些测试句子。'
     * 2296385247      'Šis ir daži pārbaudes teikumi.'
     * 3337793681      'Тестовое предложение'
     * </pre>
     * C++ code (dictionary.cc):
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * uint32_t Dictionary::hash(const std::string& str) const {
     *  uint32_t h = 2166136261;
     *  for (size_t i = 0; i < str.size(); i++) {
     *    h = h ^ uint32_t(int8_t(str[i]));
     *    h = h * 16777619;
     *  }
     *  return h;
     * }}</pre>
     * <pre> {@code
     *  uint32_t Dictionary::hash(const std::string& str) const {
     *      uint32_t h = 2166136261;
     *      for (size_t i = 0; i < str.size(); i++) {
     *          h = h ^ uint32_t(str[i]);
     *          h = h * 16777619;
     *      }
     *      return h;
     *  }
     * }</pre>
     *
     * @param str     String
     * @param charset {@link Charset}, not null
     * @return hash as long (uint32_t)
     */
    public static long hash(String str, Charset charset) {
        long h = 2_166_136_261L;// 0xffffffc5;
        for (long b : str.getBytes(charset)) {
            h = (h ^ b) * 16_777_619; // FNV-1a
        }
        return h & 0xffff_ffffL;
    }

    /**
     * @param str String
     * @return hash as long (uint32_t)
     * @see #hash(String, Charset)
     */
    long hash(String str) {
        return hash(str, charset);
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * void Dictionary::initNgrams() {
     *  for (size_t i = 0; i < size_; i++) {
     *    std::string word = BOW + words_[i].word + EOW;
     *    words_[i].subwords.clear();
     *    words_[i].subwords.push_back(i);
     *    if (words_[i].word != EOS) {
     *      computeSubwords(word, words_[i].subwords);
     *    }
     *  }
     * }}</pre>
     * <pre>{@code
     * void Dictionary::initNgrams() {
     *  for (size_t i = 0; i < size_; i++) {
     *      std::string word = BOW + words_[i].word + EOW;
     *      words_[i].subwords.clear();
     *      words_[i].subwords.push_back(i);
     *      if (words_[i].word != EOS) {
     *          computeSubwords(word, words_[i].subwords);
     *      }
     *  }
     * }}</pre>
     */
    private void initNgrams() {
        if (FastText.USE_PARALLEL_COMPUTATION && size > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, size).parallel().forEach(this::initNgrams);
            return;
        }
        for (int i = 0; i < size; i++) {
            initNgrams(i);
        }
    }

    private void initNgrams(int i) {
        Entry e = words.get(i);
        String word = BOW + e.word + EOW;
        e.subwords.clear();
        e.subwords.add(i);
        if (!EOS.equals(e.word)) {
            computeSubwords(word, e.subwords);
        }
    }

    /**
     * Original code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * void Dictionary::computeSubwords(const std::string& word, std::vector<int32_t>& ngrams) const {
     *  for (size_t i = 0; i < word.size(); i++) {
     *      std::string ngram;
     *      if ((word[i] & 0xC0) == 0x80) continue;
     *      for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
     *          ngram.push_back(word[j++]);
     *          while (j < word.size() && (word[j] & 0xC0) == 0x80) {
     *              ngram.push_back(word[j++]);
     *          }
     *          if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
     *              int32_t h = hash(ngram) % args_->bucket;
     *              pushHash(ngrams, h);
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param word String
     * @param ngrams List of ints
     */
    private void computeSubwords(String word, List<Integer> ngrams) {
        computeSubwords(word, ngrams, null, this::pushHash);
    }

    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * void Dictionary::computeSubwords(const std::string& word, std::vector<int32_t>& ngrams, std::vector<std::string>& substrings) const {
     *  for (size_t i = 0; i < word.size(); i++) {
     *      std::string ngram;
     *      if ((word[i] & 0xC0) == 0x80) continue;
     *      for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
     *          ngram.push_back(word[j++]);
     *          while (j < word.size() && (word[j] & 0xC0) == 0x80) {
     *              ngram.push_back(word[j++]);
     *          }
     *          if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
     *              int32_t h = hash(ngram) % args_->bucket;
     *              ngrams.push_back(nwords_ + h);
     *              substrings.push_back(ngram);
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param word String, the word
     * @param ngrams List of ints
     * @param substrings List of strings
     */
    private void computeSubwords(String word, List<Integer> ngrams, List<String> substrings) {
        computeSubwords(word, ngrams, substrings, (nrgams, h) -> ngrams.add(nwords + h));
    }

    private void computeSubwords(String word, List<Integer> ngrams, List<String> substrings, BiConsumer<List<Integer>, Integer> pushMethod) {
        int len = word.length();
        for (int i = 0, cpI; i < len; i += Character.charCount(cpI)) {
            cpI = word.codePointAt(i);
            StringBuilder ngram = new StringBuilder();
            for (int j = i, n = 1, cpJ; j < len && n <= maxn; n++) {
                cpJ = word.codePointAt(j);
                ngram.appendCodePoint(cpJ);
                j += Character.charCount(cpJ);
                if (n >= minn && !(n == 1 && (i == 0 || j == len))) {
                    int h = (int) (hash(ngram.toString()) % bucket.intValue());
                    pushMethod.accept(ngrams, h);
                    if (substrings != null) {
                        substrings.add(ngram.toString());
                    }
                }
            }
        }
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
     *  if (pruneidx_size_ == 0 || id < 0) {
     *    return;
     *  }
     *  if (pruneidx_size_ > 0) {
     *    if (pruneidx_.count(id)) {
     *      id = pruneidx_.at(id);
     *    } else {
     *      return;
     *    }
     *  }
     *  hashes.push_back(nwords_ + id);
     * }}</pre>
     * <pre>{@code
     * void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
     *  if (pruneidx_size_ == 0 || id < 0) return;
     *  if (pruneidx_size_ > 0) {
     *      if (pruneidx_.count(id)) {
     *          id = pruneidx_.at(id);
     *      } else {
     *          return;
     *      }
     *  }
     *  hashes.push_back(nwords_ + id);
     * }
     * }</pre>
     *
     * @param hashes List of ints
     * @param id int
     */
    private void pushHash(List<Integer> hashes, int id) {
        if (pruneIdxSize == 0 || id < 0) return;
        if (pruneIdxSize > 0) {
            if (pruneIdx.containsKey(id)) {
                id = pruneIdx.get(id);
            } else {
                return;
            }
        }
        hashes.add(nwords + id);
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code void Dictionary::initTableDiscard() {
     *  pdiscard_.resize(size_);
     *  for (size_t i = 0; i < size_; i++) {
     *    real f = real(words_[i].count) / real(ntokens_);
     *    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
     *  }
     * }}</pre>
     * <pre>{@code void Dictionary::initTableDiscard() {
     *  pdiscard_.resize(size_);
     *  for (size_t i = 0; i < size_; i++) {
     *      real f = real(words_[i].count) / real(ntokens_);
     *      pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
     *  }
     * }}</pre>
     */
    private void initTableDiscard() {
        pdiscard = Floats.asList(new float[size]);
        if (FastText.USE_PARALLEL_COMPUTATION && size > PARALLEL_SIZE_THRESHOLD) {
            IntStream.range(0, size).parallel().forEach(i -> {
                float f = ((float) words.get(i).count) / ntokens;
                pdiscard.set(i, (float) (FastMath.sqrt(t / f) + t / f));
            });
            return;
        }
        for (int i = 0; i < size; i++) {
            float f = ((float) words.get(i).count) / ntokens;
            pdiscard.set(i, (float) (FastMath.sqrt(t / f) + t / f));
        }
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
     *  std::vector<int64_t> counts;
     *  for (auto& w : words_) {
     *    if (w.type == type) {
     *      counts.push_back(w.count);
     *    }
     *  }
     *  return counts;
     * }}</pre>
     * <pre>{@code
     * std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
     *  std::vector<int64_t> counts;
     *  for (auto& w : words_) {
     *      if (w.type == type) counts.push_back(w.count);
     *  }
     *  return counts;
     * }
     * }</pre>
     *
     * @param type {@link EntryType}
     * @return List of longs
     */
    public List<Long> getCounts(EntryType type) {
        List<Long> counts = new ArrayList<>(EntryType.LABEL == type ? nlabels() : nwords());
        for (Entry w : words) {
            if (w.type == type)
                counts.add(w.count);
        }
        return counts;
    }


    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * int32_t Dictionary::getLine(std::istream& in, std::vector<int32_t>& words, std::vector<int32_t>& labels, std::minstd_rand& rng) const {
     *  std::vector<int32_t> word_hashes;
     *  std::string token;
     *  int32_t ntokens = 0;
     *  reset(in);
     *  words.clear();
     *  labels.clear();
     *  while (readWord(in, token)) {
     *      uint32_t h = hash(token);
     *      int32_t wid = getId(token, h);
     *      entry_type type = wid < 0 ? getType(token) : getType(wid);
     *      ntokens++;
     *      if (type == entry_type::word) {
     *          addSubwords(words, token, wid);
     *          word_hashes.push_back(h);
     *      } else if (type == entry_type::label && wid >= 0) {
     *          labels.push_back(wid - nwords_);
     *      }
     *      if (token == EOS) break;
     *  }
     *  addWordNgrams(words, word_hashes, args_->wordNgrams);
     *  return ntokens;
     * }
     * }</pre>
     *
     * @param in     {@link SeekableReader}
     * @param words  List of words
     * @param labels List of labels
     * @return int32_t
     * @throws IOException if an I/O error occurs
     */
    int getLine(SeekableReader in, List<Integer> words, List<Integer> labels) throws IOException {
        in.rewind();
        List<Integer> wordHashes = new ArrayList<>();
        int ntokens = 0;
        words.clear();
        labels.clear();
        String token;
        while ((token = in.nextWord()) != null) {
            ntokens++;
            long h = hash(token);
            int wid = getId(token, h);
            EntryType type = wid < 0 ? getType(token) : getType(wid);
            if (EntryType.WORD == type) {
                addSubwords(words, token, wid);
                wordHashes.add((int) h);
            } else if (EntryType.LABEL == type && wid >= 0) {
                labels.add(wid - nwords);
            }
            if (Objects.equals(token, EOS)) {
                break;
            }
        }
        addWordNgrams(words, wordHashes);
        return ntokens;
    }

    public List<Integer> getLine(String line) {
        List<Integer> res = new ArrayList<>();
        InputStream in = new ByteArrayInputStream(line.getBytes(charset));
        try {
            getLine(createReader(in), res, new ArrayList<>());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return res;
    }

    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * int32_t Dictionary::getLine(std::istream& in, std::vector<int32_t>& words, std::minstd_rand& rng) const {
     *  std::uniform_real_distribution<> uniform(0, 1);
     *  std::string token;
     *  int32_t ntokens = 0;
     *  reset(in);
     *  words.clear();
     *  while (readWord(in, token)) {
     *      int32_t h = find(token);
     *      int32_t wid = word2int_[h];
     *      if (wid < 0) continue;
     *      ntokens++;
     *      if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
     *          words.push_back(wid);
     *      }
     *      if (ntokens > MAX_LINE_SIZE || token == EOS) break;
     *  }
     *  return ntokens;
     * }
     * }</pre>
     *
     * @param in    {@link SeekableReader}
     * @param words List of words
     * @param rng   {@link RandomGenerator}
     * @return int32_t
     * @throws IOException if an I/O error occurs
     */
    int getLine(SeekableReader in, List<Integer> words, RandomGenerator rng) throws IOException {
        in.rewind();
        UniformRealDistribution uniform = new UniformRealDistribution(rng, 0, 1);
        int ntokens = 0;
        words.clear();
        String token;
        while ((token = in.nextWord()) != null) {
            long h = hash(token);
            int wid = getId(token, h);
            if (wid < 0) continue;
            ntokens++;
            if (EntryType.WORD == getType(wid) && !discard(wid, uniform.sample())) {
                words.add(wid);
            }
            if (ntokens > MAX_LINE_SIZE || Objects.equals(token, EOS)) break;
        }
        return ntokens;
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * bool Dictionary::discard(int32_t id, real rand) const {
     *  assert(id >= 0);
     *  assert(id < nwords_);
     *  if (args_->model == model_name::sup) {
     *    return false;
     *  }
     *  return rand > pdiscard_[id];
     * }}</pre>
     * <pre>{@code
     * bool Dictionary::discard(int32_t id, real rand) const {
     *  assert(id >= 0);
     *  assert(id < nwords_);
     *  if (args_->model == model_name::sup) return false;
     *      return rand > pdiscard_[id];
     * }
     * }</pre>
     *
     * @param id int
     * @param rand rand
     * @return boolean
     */
    private boolean discard(int id, double rand) {
        Validate.isTrue(id >= 0);
        Validate.isTrue(id < nwords);
        return model != Args.ModelName.SUP && rand > pdiscard.get(id);
    }

    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * void Dictionary::addWordNgrams(std::vector<int32_t>& line, const std::vector<int32_t>& hashes, int32_t n) const {
     *  for (int32_t i = 0; i < hashes.size(); i++) {
     *      uint64_t h = hashes[i];
     *      for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
     *          h = h * 116049371 + hashes[j];
     *          pushHash(line, h % args_->bucket);
     *      }
     *  }
     * }
     * }</pre>
     *
     * @param line   List of ints
     * @param hashes List of ints
     * @param n      int
     */
    private void addWordNgrams(List<Integer> line, List<Integer> hashes, int n) {
        if (FastText.USE_PARALLEL_COMPUTATION && hashes.size() > PARALLEL_SIZE_THRESHOLD) {
            List<Integer> sync = Collections.synchronizedList(line);
            IntStream.range(0, hashes.size()).parallel().forEach(i -> addWordNgrams(sync, hashes, i, n));
            return;
        }
        for (int i = 0; i < hashes.size(); i++) { // int32_t
            addWordNgrams(line, hashes, i, n);
        }
    }

    private void addWordNgrams(List<Integer> line, List<Integer> hashes, int i, int n) {
        UnsignedLong h = UnsignedLong.fromLongBits(hashes.get(i)); // uint64_t
        for (int j = i + 1; j < hashes.size() && j < i + n; j++) { // h = h * 116049371 + hashes[j] :
            h = h.times(ADD_WORDS_NGRAMS_FACTOR_UNSIGNED_LONG).plus(UnsignedLong.fromLongBits(hashes.get(j)));
            pushHash(line, h.mod(bucket).intValue()); // h % args_->bucket
        }
    }

    private void addWordNgrams(List<Integer> line, List<Integer> hashes) {
        addWordNgrams(line, hashes, wordNgrams);
    }

    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * void Dictionary::addSubwords(std::vector<int32_t>& line, const std::string& token, int32_t wid) const {
     *  if (wid < 0) { // out of vocab
     *      computeSubwords(BOW + token + EOW, line);
     *  } else {
     *      if (args_->maxn <= 0) { // in vocab w/o subwords
     *          line.push_back(wid);
     *      } else { // in vocab w/ subwords
     *          const std::vector<int32_t>& ngrams = getSubwords(wid);
     *          line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
     *      }
     * }
     * }}</pre>
     *
     * @param line List of ints
     * @param token String token
     * @param wid int, word id
     */
    private void addSubwords(List<Integer> line, String token, int wid) {
        if (wid < 0) { // out of vocab
            computeSubwords(BOW + token + EOW, line);
        } else {
            if (maxn <= 0) { // in vocab w/o subwords
                line.add(wid);
            } else { // in vocab w/ subwords
                line.addAll(getSubwords(wid));
            }
        }
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
     *  assert(i >= 0);
     *  assert(i < nwords_);
     *  return words_[i].subwords;
     * }}</pre>
     * <pre>{@code
     * const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
     *  assert(i >= 0);
     *  assert(i < nwords_);
     *  return words_[i].subwords;
     * }
     * }</pre>
     *
     * @param i int
     * @return List of ints
     */
    public List<Integer> getSubwords(int i) {
        Validate.isTrue(i >= 0);
        Validate.isTrue(i < nwords);
        return words.get(i).subwords;
    }

    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * const std::vector<int32_t> Dictionary::getSubwords(const std::string& word) const {
     *  int32_t i = getId(word);
     *  if (i >= 0) {
     *      return getSubwords(i);
     *  }
     *  std::vector<int32_t> ngrams;
     *  computeSubwords(BOW + word + EOW, ngrams);
     *  return ngrams;
     * }}</pre>
     *
     * @param word String
     * @return List of ints
     */
    public List<Integer> getSubwords(String word) {
        int i = getId(word);
        if (i >= 0) {
            return getSubwords(i);
        }
        List<Integer> ngrams = new ArrayList<>();
        computeSubwords(BOW + word + EOW, ngrams);
        return ngrams;
    }

    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code
     * void Dictionary::getSubwords(const std::string& word, std::vector<int32_t>& ngrams, std::vector<std::string>& substrings) const {
     *  int32_t i = getId(word);
     *  ngrams.clear();
     *  substrings.clear();
     *  if (i >= 0) {
     *      ngrams.push_back(i);
     *      substrings.push_back(words_[i].word);
     *  } else {
     *      ngrams.push_back(-1);
     *      substrings.push_back(word);
     *  }
     *  computeSubwords(BOW + word + EOW, ngrams, substrings);
     * }}</pre>
     *
     * @param word String
     * @return {@link Multimap}
     */
    public Multimap<String, Integer> getSubwordsMap(String word) {
        List<Integer> ngrams = new ArrayList<>();
        List<String> substrings = new ArrayList<>();
        int i = getId(word);
        if (i >= 0) {
            ngrams.add(i);
            substrings.add(words.get(i).word);
        } else {
            ngrams.add(-1);
            substrings.add(word);
        }
        computeSubwords(BOW + word + EOW, ngrams, substrings);
        if (ngrams.size() != substrings.size()) {
            throw new IllegalStateException("ngrams(" + ngrams.size() + ") != substrings(" + substrings.size() + ")");
        }
        Multimap<String, Integer> res = ArrayListMultimap.create(ngrams.size(), 1);
        for (int j = 0; j < ngrams.size(); j++) {
            res.put(substrings.get(j), ngrams.get(j));
        }
        return res;
    }

    /**
     * Creates a dictionary Reader instance.
     *
     * @param in {@link InputStream}
     * @return {@link SeekableReader}
     * @see #createWordReader(InputStream, Charset, int)
     * @see #createSeekableWordReader(InputStream, Charset, int)
     */
    public SeekableReader createReader(InputStream in) {
        return createSeekableWordReader(in, charset, FastText.Factory.BUFF_SIZE);
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * std::string Dictionary::getLabel(int32_t lid) const {
     *  if (lid < 0 || lid >= nlabels_) {
     *    throw std::invalid_argument(
     *        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
     *  }
     *  return words_[lid + nwords_].word;
     * }}</pre>
     * <pre>{@code
     * std::string Dictionary::getLabel(int32_t lid) const {
     *  if (lid < 0 || lid >= nlabels_) {
     *      throw std::invalid_argument("Label id is out of range [0, " + std::to_string(nlabels_) + "]");
     *  }
     *  return words_[lid + nwords_].word;
     * }}</pre>
     *
     * @param lid int32_t
     * @return String, label
     */
    public String getLabel(int lid) {
        if (lid < 0 || lid >= nlabels) {
            throw new IllegalArgumentException("Label id is out of range [0, " + nlabels + "]");
        }
        return words.get(lid + nwords).word;
    }

    /**
     * Original (c++) code:
     * // FIXME: The original code could not be found by auto completion.
     * <pre>{@code bool isPruned() {
     *  return pruneidx_size_ >= 0;
     *  }
     * }</pre>
     *
     * @return true if pruned
     */
    public boolean isPruned() {
        return pruneIdxSize >= 0;
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * void Dictionary::threshold(int64_t t, int64_t tl) {
     *  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
     *    if (e1.type != e2.type) {
     *      return e1.type < e2.type;
     *    }
     *    return e1.count > e2.count;
     *  });
     *  words_.erase(
     *      remove_if(
     *          words_.begin(),
     *          words_.end(),
     *          [&](const entry& e) {
     *            return (e.type == entry_type::word && e.count < t) ||
     *                (e.type == entry_type::label && e.count < tl);
     *          }),
     *      words_.end());
     *  words_.shrink_to_fit();
     *  size_ = 0;
     *  nwords_ = 0;
     *  nlabels_ = 0;
     *  std::fill(word2int_.begin(), word2int_.end(), -1);
     *  for (auto it = words_.begin(); it != words_.end(); ++it) {
     *    int32_t h = find(it->word);
     *    word2int_[h] = size_++;
     *    if (it->type == entry_type::word) {
     *      nwords_++;
     *    }
     *    if (it->type == entry_type::label) {
     *      nlabels_++;
     *    }
     *  }
     * }}</pre>
     * <pre>{@code
     * void Dictionary::threshold(int64_t t, int64_t tl) {
     *  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
     *      if (e1.type != e2.type) return e1.type < e2.type;
     *      return e1.count > e2.count;
     *  });
     *  words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
     *      return (e.type == entry_type::word && e.count < t) || (e.type == entry_type::label && e.count < tl);
     *  }), words_.end());
     *  words_.shrink_to_fit();
     *  size_ = 0;
     *  nwords_ = 0;
     *  nlabels_ = 0;
     *  std::fill(word2int_.begin(), word2int_.end(), -1);
     *  for (auto it = words_.begin(); it != words_.end(); ++it) {
     *      int32_t h = find(it->word);
     *      word2int_[h] = size_++;
     *      if (it->type == entry_type::word) nwords_++;
     *      if (it->type == entry_type::label) nlabels_++;
     *  }
     * }
     * }</pre>
     *
     * @param wordThreshold long
     * @param labelThreshold long
     */
    void threshold(long wordThreshold, long labelThreshold) {
        Stream<Entry> entries = this.words.stream()
                .filter(e -> (EntryType.WORD != e.type || e.count >= wordThreshold) && (EntryType.LABEL != e.type || e.count >= labelThreshold))
                .sorted(ENTRY_COMPARATOR);
        if (FastText.USE_PARALLEL_COMPUTATION && this.size > PARALLEL_SIZE_THRESHOLD) {
            entries = entries.parallel();
        }
        ArrayList<Entry> words = entries.collect(Collectors.toCollection(ArrayList::new));
        words.trimToSize();
        this.words = words;
        this.word2int = new HashMap<>(words.size());
        int wordsCount = 0;
        int labelsCount = 0;
        int count = 0;
        for (Entry e : words) {
            long h = find(e.word);
            word2int.put(h, count++);
            if (EntryType.WORD == e.type) wordsCount++;
            if (EntryType.LABEL == e.type) labelsCount++;
        }
        this.size = this.words.size();
        this.nwords = wordsCount;
        this.nlabels = labelsCount;
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code void Dictionary::prune(std::vector<int32_t>& idx) {
     *  std::vector<int32_t> words, ngrams;
     *  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
     *    if (*it < nwords_) {
     *      words.push_back(*it);
     *    } else {
     *      ngrams.push_back(*it);
     *    }
     *  }
     *  std::sort(words.begin(), words.end());
     *  idx = words;
     * 
     *  if (ngrams.size() != 0) {
     *    int32_t j = 0;
     *    for (const auto ngram : ngrams) {
     *      pruneidx_[ngram - nwords_] = j;
     *      j++;
     *    }
     *    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
     *  }
     *  pruneidx_size_ = pruneidx_.size();
     * 
     *  std::fill(word2int_.begin(), word2int_.end(), -1);
     * 
     *  int32_t j = 0;
     *  for (int32_t i = 0; i < words_.size(); i++) {
     *    if (getType(i) == entry_type::label ||
     *        (j < words.size() && words[j] == i)) {
     *      words_[j] = words_[i];
     *      word2int_[find(words_[j].word)] = j;
     *      j++;
     *    }
     *  }
     *  nwords_ = words.size();
     *  size_ = nwords_ + nlabels_;
     *  words_.erase(words_.begin() + size_, words_.end());
     *  initNgrams();
     * }}</pre>
     * <pre>{@code void Dictionary::prune(std::vector<int32_t>& idx) {
     *  std::vector<int32_t> words, ngrams;
     *  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
     *      if (*it < nwords_) {
     *          words.push_back(*it);
     *      } else {
     *          ngrams.push_back(*it);
     *      }
     *  }
     *  std::sort(words.begin(), words.end());
     *  idx = words;
     *  if (ngrams.size() != 0) {
     *      int32_t j = 0;
     *      for (const auto ngram : ngrams) {
     *          pruneidx_[ngram - nwords_] = j;
     *          j++;
     *      }
     *      idx.insert(idx.end(), ngrams.begin(), ngrams.end());
     *  }
     *  pruneidx_size_ = pruneidx_.size();
     *  std::fill(word2int_.begin(), word2int_.end(), -1);
     *  int32_t j = 0;
     *  for (int32_t i = 0; i < words_.size(); i++) {
     *      if (getType(i) == entry_type::label || (j < words.size() && words[j] == i)) {
     *          words_[j] = words_[i];
     *          word2int_[find(words_[j].word)] = j;
     *          j++;
     *      }
     *  }
     *  nwords_ = words.size();
     *  size_ = nwords_ +  nlabels_;
     *  words_.erase(words_.begin() + size_, words_.end());
     *  initNgrams();
     * }}</pre>
     *
     * @param idx List of ints
     * @return List of ints
     */
    List<Integer> prune(List<Integer> idx) {
        List<Integer> words = new ArrayList<>();
        List<Integer> ngrams = new ArrayList<>();
        for (Integer it : idx) {
            if (it < nwords) {
                words.add(it);
            } else {
                ngrams.add(it);
            }
        }
        Collections.sort(words);
        List<Integer> res = new ArrayList<>(words);
        if (!ngrams.isEmpty()) {
            int j = 0;
            for (int ngram : ngrams) {
                pruneIdx.put(ngram - nwords, j);
                j++;
            }
            res.addAll(ngrams);
        }
        pruneIdxSize = pruneIdx.size();
        word2int.clear();
        int j = 0;
        for (int i = 0; i < this.words.size(); i++) {
            if (getType(i) != EntryType.LABEL && (j >= words.size() || words.get(j) != i)) {
                continue;
            }
            this.words.set(j, this.words.get(i));
            word2int.put(find(this.words.get(j).word), j);
            j++;
        }
        nwords = words.size();
        size = nwords + nlabels;
        this.words = this.words.subList(0, size);
        initNgrams();
        return res;
    }

    /**
     * Makes a full (deep) copy of the instance
     *
     * @return {@link Dictionary}
     */
    public Dictionary copy() {
        Dictionary res = new Dictionary(model, label, t, bucket, maxn, minn, wordNgrams, charset);
        res.size = this.size;
        res.nwords = this.nwords;
        res.nlabels = this.nlabels;
        res.ntokens = this.ntokens;
        res.pruneIdxSize = this.pruneIdxSize;
        res.word2int = new HashMap<>(this.word2int);
        res.words = new ArrayList<>(this.words.size());
        this.words.forEach(entry -> res.words.add(entry.copy()));
        res.words = new ArrayList<>(this.words);
        res.pruneIdx = new HashMap<>(this.pruneIdx);
        res.pdiscard = Floats.asList(Floats.toArray(this.pdiscard));
        return res;
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code
     * void Dictionary::save(std::ostream& out) const {
     *  out.write((char*)&size_, sizeof(int32_t));
     *  out.write((char*)&nwords_, sizeof(int32_t));
     *  out.write((char*)&nlabels_, sizeof(int32_t));
     *  out.write((char*)&ntokens_, sizeof(int64_t));
     *  out.write((char*)&pruneidx_size_, sizeof(int64_t));
     *  for (int32_t i = 0; i < size_; i++) {
     *    entry e = words_[i];
     *    out.write(e.word.data(), e.word.size() * sizeof(char));
     *    out.put(0);
     *    out.write((char*)&(e.count), sizeof(int64_t));
     *    out.write((char*)&(e.type), sizeof(entry_type));
     *  }
     *  for (const auto pair : pruneidx_) {
     *    out.write((char*)&(pair.first), sizeof(int32_t));
     *    out.write((char*)&(pair.second), sizeof(int32_t));
     *  }
     * }}</pre>
     * <pre>{@code
     * void Dictionary::save(std::ostream& out) const {
     *  out.write((char*) &size_, sizeof(int32_t));
     *  out.write((char*) &nwords_, sizeof(int32_t));
     *  out.write((char*) &nlabels_, sizeof(int32_t));
     *  out.write((char*) &ntokens_, sizeof(int64_t));
     *  out.write((char*) &pruneidx_size_, sizeof(int64_t));
     *  for (int32_t i = 0; i < size_; i++) {
     *      entry e = words_[i];
     *      out.write(e.word.data(), e.word.size() * sizeof(char));
     *      out.put(0);
     *      out.write((char*) &(e.count), sizeof(int64_t));
     *      out.write((char*) &(e.type), sizeof(entry_type));
     *  }
     *  for (const auto pair : pruneidx_) {
     *      out.write((char*) &(pair.first), sizeof(int32_t));
     *      out.write((char*) &(pair.second), sizeof(int32_t));
     *  }
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    void save(FTOutputStream out) throws IOException {
        out.writeInt(size);
        out.writeInt(nwords);
        out.writeInt(nlabels);
        out.writeLong(ntokens);
        out.writeLong(pruneIdxSize);
        for (Entry e : words) {
            FTOutputStream.writeString(out, e.word, charset);
            out.writeLong(e.count);
            out.writeByte(e.type.ordinal());
        }
        for (Integer key : pruneIdx.keySet()) {
            out.writeInt(key);
            out.writeInt(pruneIdx.get(key));
        }
    }

    /**
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code void Dictionary::load(std::istream& in) {
     *  words_.clear();
     *  in.read((char*)&size_, sizeof(int32_t));
     *  in.read((char*)&nwords_, sizeof(int32_t));
     *  in.read((char*)&nlabels_, sizeof(int32_t));
     *  in.read((char*)&ntokens_, sizeof(int64_t));
     *  in.read((char*)&pruneidx_size_, sizeof(int64_t));
     *  for (int32_t i = 0; i < size_; i++) {
     *    char c;
     *    entry e;
     *    while ((c = in.get()) != 0) {
     *      e.word.push_back(c);
     *    }
     *    in.read((char*)&e.count, sizeof(int64_t));
     *    in.read((char*)&e.type, sizeof(entry_type));
     *    words_.push_back(e);
     *  }
     *  pruneidx_.clear();
     *  for (int32_t i = 0; i < pruneidx_size_; i++) {
     *    int32_t first;
     *    int32_t second;
     *    in.read((char*)&first, sizeof(int32_t));
     *    in.read((char*)&second, sizeof(int32_t));
     *    pruneidx_[first] = second;
     *  }
     *  initTableDiscard();
     *  initNgrams();
     * 
     *  int32_t word2intsize = std::ceil(size_ / 0.7);
     *  word2int_.assign(word2intsize, -1);
     *  for (int32_t i = 0; i < size_; i++) {
     *    word2int_[find(words_[i].word)] = i;
     *  }
     * }}</pre>
     * <pre>{@code void Dictionary::load(std::istream& in) {
     *  words_.clear();
     *  std::fill(word2int_.begin(), word2int_.end(), -1);
     *  in.read((char*) &size_, sizeof(int32_t));
     *  in.read((char*) &nwords_, sizeof(int32_t));
     *  in.read((char*) &nlabels_, sizeof(int32_t));
     *  in.read((char*) &ntokens_, sizeof(int64_t));
     *  in.read((char*) &pruneidx_size_, sizeof(int64_t));
     *  for (int32_t i = 0; i < size_; i++) {
     *      char c;
     *      entry e;
     *      while ((c = in.get()) != 0) {
     *          e.word.push_back(c);
     *      }
     *      in.read((char*) &e.count, sizeof(int64_t));
     *      in.read((char*) &e.type, sizeof(entry_type));
     *      words_.push_back(e);
     *      word2int_[find(e.word)] = i;
     *  }
     *  pruneidx_.clear();
     *  for (int32_t i = 0; i < pruneidx_size_; i++) {
     *      int32_t first;
     *      int32_t second;
     *      in.read((char*) &first, sizeof(int32_t));
     *      in.read((char*) &second, sizeof(int32_t));
     *      pruneidx_[first] = second;
     *  }
     *  initTableDiscard();
     *  initNgrams();
     * }}</pre>
     *
     * @param args {@link Args}
     * @param charset {@link Charset}
     * @param in      {@link FTInputStream}
     * @return {@link Dictionary} new instance
     * @throws IOException if an I/O error occurs
     */
    static Dictionary load(Args args, Charset charset, FTInputStream in) throws IOException {
        Dictionary res = new Dictionary(args, charset);
        res.size = in.readInt();
        res.nwords = in.readInt();
        res.nlabels = in.readInt();
        res.ntokens = in.readLong();
        res.pruneIdxSize = in.readLong();
        res.word2int = new HashMap<>(res.size);
        res.words = new ArrayList<>(res.size);
        for (int i = 0; i < res.size; i++) {
            Entry e = new Entry(FTInputStream.readString(in, res.charset), in.readLong(), EntryType.fromValue(in.readByte()));
            res.words.add(e);
            res.word2int.put(res.find(e.word), i);
        }
        res.pruneIdx.clear();
        for (int i = 0; i < res.pruneIdxSize; i++) {
            res.pruneIdx.put(in.readInt(), in.readInt());
        }
        res.initTableDiscard();
        res.initNgrams();
        return res;
    }

    /**
     * Reads a dictionary from stream.
     * Original (c++) code:
     * // FIXME: Auto completion found the original code. Check out differences.
     * <pre>{@code void Dictionary::readFromFile(std::istream& in) {
     *  std::string word;
     *  int64_t minThreshold = 1;
     *  while (readWord(in, word)) {
     *    add(word);
     *    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
     *      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
     *    }
     *    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
     *      minThreshold++;
     *      threshold(minThreshold, minThreshold);
     *    }
     *  }
     *  threshold(args_->minCount, args_->minCountLabel);
     *  initTableDiscard();
     *  initNgrams();
     *  if (args_->verbose > 0) {
     *    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
     *    std::cerr << "Number of words:  " << nwords_ << std::endl;
     *    std::cerr << "Number of labels: " << nlabels_ << std::endl;
     *  }
     *  if (size_ == 0) {
     *    throw std::invalid_argument(
     *        "Empty vocabulary. Try a smaller -minCount value.");
     *  }
     * }}</pre>
     * <pre>{@code void Dictionary::readFromFile(std::istream& in) {
     *  std::string word;
     *  int64_t minThreshold = 1;
     *  while (readWord(in, word)) {
     *      add(word);
     *      if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
     *          std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
     *      }
     *      if (size_ > 0.75 * MAX_VOCAB_SIZE) {
     *          minThreshold++;
     *          threshold(minThreshold, minThreshold);
     *      }
     *  }
     *  threshold(args_->minCount, args_->minCountLabel);
     *  initTableDiscard();
     *  initNgrams();
     *  if (args_->verbose > 0) {
     *      std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
     *      std::cerr << "Number of words:  " << nwords_ << std::endl;
     *      std::cerr << "Number of labels: " << nlabels_ << std::endl;
     *  }
     *  if (size_ == 0) {
     *      std::cerr << "Empty vocabulary. Try a smaller -minCount value." << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     * }}</pre>
     *
     * @param in {@link InputStream}
     * @param args {@link Args}
     * @param charset {@link Charset}
     * @param logs {@link PrintLogs} to log process
     * @return {@link Dictionary}
     * @throws IOException in case of error with stream
     * @throws IllegalStateException if no words in dictionary
     */
    public static Dictionary read(InputStream in, Args args, Charset charset, PrintLogs logs)
            throws IOException, IllegalStateException {
        WordReader reader = createWordReader(in, charset, FastText.Factory.BUFF_SIZE);
        Dictionary res = new Dictionary(args, charset);

        long minThreshold = 1;
        String word;

        while ((word = reader.nextWord()) != null) {
            res.add(word);
            if (logs.isDebugEnabled() && res.ntokens % READ_LOG_STEP == 0) {
                logs.debug("\rRead %dM words", res.ntokens / READ_LOG_STEP);
            }
            if (res.size > 0.75 * MAX_VOCAB_SIZE) {
                minThreshold++;
                res.threshold(minThreshold, minThreshold);
            }
        }
        res.threshold(args.minCount(), args.minCountLabel());
        res.initTableDiscard();
        res.initNgrams();
        logs.infoln("\rRead %dM words", res.ntokens / READ_LOG_STEP);
        logs.infoln("Number of words:  %d", res.nwords);
        logs.infoln("Number of labels: %d", res.nlabels);
        if (res.size == 0) {
            throw new IllegalStateException("Empty vocabulary. Try a smaller -minCount value.");
        }
        return res;
    }

    /**
     * Creates a word reader.
     *
     * @param in       {@link InputStream}
     * @param charset  {@link Charset}
     * @param buffSize int buffer size
     * @return {@link WordReader}
     */
    public static WordReader createWordReader(InputStream in, Charset charset, int buffSize) {
        return new WordReader(in, charset, buffSize, EOS, DELIMITERS);
    }

    /**
     * Creates a word reader with seek supporting
     *
     * @param in       {@link InputStream}
     * @param charset  {@link Charset}
     * @param buffSize int buffer size
     * @return {@link SeekableReader}
     */
    public static SeekableReader createSeekableWordReader(InputStream in, Charset charset, int buffSize) {
        return new SeekableReader(in, charset, buffSize, EOS, DELIMITERS);
    }

    /**
     * The seekable word reader.
     *
     * @see WordReader
     */
    public static class SeekableReader extends WordReader {

        public SeekableReader(InputStream in, Charset charset, int bufferSize, String newLineSymbol, String delimiters) {
            super(in, charset, bufferSize, newLineSymbol, delimiters);
        }

        /**
         * @return true on end of stream
         */
        @Override
        public boolean isEnd() {
            return super.isEnd();
        }

        /**
         * Seeks to the given offset in bytes from the start of the stream.
         *
         * @param n number of bytes
         * @throws IOException                   if an I/O error occurs
         * @throws UnsupportedOperationException if this operation is not supported by the underlying stream
         * @throws IllegalStateException         if the result position is wrong after seek operation
         */
        public void seek(long n) throws IOException, UnsupportedOperationException, IllegalStateException {
            checkIsSeekable();
            doSeek(n);
            if (n == ((ScrollableInputStream) in).getPos()) {
                super.reset();
                return;
            }
            throw new IllegalStateException("Can't seek to " + n + " position.");
        }

        /**
         * Resets stream to the start position.

         * @return true if stream has been reset to initial zero position
         * @throws IOException                   if an I/O error occurs
         * @throws UnsupportedOperationException if this operation is not supported by the underlying stream
         */
        public boolean rewind() throws IOException, UnsupportedOperationException {
            if (!isEnd()) return false;
            checkIsSeekable();
            doSeek(0);
            return true;
        }

        private void checkIsSeekable() {
            if (in instanceof ScrollableInputStream) {
                return;
            }
            throw new UnsupportedOperationException("Encapsulated stream is not seekable.");
        }

        private void doSeek(long n) throws IOException {
            ((ScrollableInputStream) in).seek(n);
            super.reset();
        }
    }

    public enum EntryType {
        WORD, LABEL;

        public static EntryType fromValue(int value) throws IllegalArgumentException {
            return Arrays.stream(values())
                    .filter(v -> v.ordinal() == value)
                    .findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("Unknown entry_type enum value: " + value));
        }
    }

    public static class Entry {
        final String word;
        final EntryType type;
        final List<Integer> subwords = new ArrayList<>();
        long count;

        private Entry(String word, long count, EntryType type) {
            this.word = word;
            this.count = count;
            this.type = type;
        }

        @Override
        public String toString() {
            return String.format("entry [word=%s, count=%d, type=%s, subwords=%s]", word, count, type, subwords);
        }

        public long count() {
            return count;
        }

        Entry copy() {
            Entry res = new Entry(this.word, this.count, this.type);
            res.subwords.addAll(this.subwords);
            return res;
        }
    }

}
