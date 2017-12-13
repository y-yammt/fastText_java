package cc.fasttext;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.Charset;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.ToLongFunction;
import java.util.stream.Collectors;

import org.apache.commons.lang.Validate;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;

import cc.fasttext.io.FTInputStream;
import cc.fasttext.io.FTOutputStream;
import cc.fasttext.io.FTReader;
import cc.fasttext.io.PrintLogs;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.primitives.UnsignedLong;

/**
 * See <a href='https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc'>dictionary.cc</a> &
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h'>dictionary.h</a>
 */
public strictfp class Dictionary {

    private static final int MAX_VOCAB_SIZE = 30_000_000;
    private static final int MAX_LINE_SIZE = 1024;
    private static final Integer WORDID_DEFAULT = -1;
    private static final Integer PRUNE_IDX_SIZE_DEFAULT = -1;

    public static final String EOS = "</s>";
    public static final String BOW = "<";
    public static final String EOW = ">";

    private static final Comparator<Entry> ENTRY_COMPARATOR = Comparator.comparing((Function<Entry, EntryType>) t -> t.type)
            .thenComparing(Comparator.comparingLong((ToLongFunction<Entry>) value -> value.count).reversed());
    private List<Entry> words_;
    private List<Float> pdiscard_;
    private Map<Long, Integer> word2int_;
    private int size_;
    private int nwords_;
    private int nlabels_;
    private long ntokens_;
    private long pruneidx_size_ = PRUNE_IDX_SIZE_DEFAULT;
    private Map<Integer, Integer> pruneidx_ = new HashMap<>();
    private final Args args;
    private final Charset charset;

    Dictionary(Args args, Charset charset) {
        this.args = args;
        this.charset = charset;
        word2int_ = new HashMap<>(MAX_VOCAB_SIZE);
        words_ = new ArrayList<>(MAX_VOCAB_SIZE);
    }

    public Charset charset() {
        return charset;
    }

    long find(String w) {
        return find(w, hash(w));
    }

    /**
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
        while (!Objects.equals(word2int_.getOrDefault(id, WORDID_DEFAULT), WORDID_DEFAULT) &&
                !Objects.equals(words_.get(word2int_.get(id)).word, w)) {
            id = (id + 1) % MAX_VOCAB_SIZE;
        }
        return id;
    }

    /**
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
        ntokens_++;
        if (Objects.equals(word2int_.getOrDefault(h, WORDID_DEFAULT), WORDID_DEFAULT)) {
            Entry e = new Entry(w, 1, getType(w));
            words_.add(e);
            word2int_.put(h, size_++);
        } else {
            words_.get(word2int_.get(h)).count++;
        }
    }

    public int nwords() {
        return nwords_;
    }

    public int nlabels() {
        return nlabels_;
    }

    public long ntokens() {
        return ntokens_;
    }

    public List<Entry> getWords() {
        return words_;
    }

    public List<Float> getPdiscard() {
        return pdiscard_;
    }

    Map<Long, Integer> getWord2int() {
        return word2int_;
    }

    public int getSize() {
        return size_;
    }

    /**
     * <pre>{@code int32_t Dictionary::getId(const std::string& w) const {
     *  int32_t h = find(w);
     *  return word2int_[h];
     * }}</pre>
     *
     * @param w String
     * @return int32_t
     */
    public int getId(String w) {
        return word2int_.getOrDefault(find(w), WORDID_DEFAULT);
    }

    /**
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
        return word2int_.getOrDefault(find(w, h), WORDID_DEFAULT);
    }

    /**
     * <pre>{@code entry_type Dictionary::getType(const std::string& w) const {
     *  return (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
     * }}</pre>
     *
     * @param w String
     * @return {@link EntryType}
     */
    private EntryType getType(String w) {
        return w.startsWith(args.label()) ? EntryType.LABEL : EntryType.WORD;
    }

    /**
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
        Validate.isTrue(id < size_);
        return words_.get(id).type;
    }

    public String getWord(int id) {
        Validate.isTrue(id >= 0);
        Validate.isTrue(id < size_);
        return words_.get(id).word;
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
     * @param str String
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
        for (int i = 0; i < size_; i++) {
            String word = BOW + words_.get(i).word + EOW;
            Entry e = words_.get(i);
            e.subwords.clear();
            e.subwords.add(i);
            if (!EOS.equals(e.word)) {
                computeSubwords(word, e.subwords);
            }
        }
    }

    /**
     * Original code:
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
     * @param word
     * @param ngrams
     */
    private void computeSubwords(String word, List<Integer> ngrams) {
        computeSubwords(word, ngrams, null, this::pushHash);
    }

    /**
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
     * @param word
     * @param ngrams
     * @param substrings
     */
    private void computeSubwords(String word, List<Integer> ngrams, List<String> substrings) {
        computeSubwords(word, ngrams, substrings, (nrgams, h) -> ngrams.add(nwords_ + h));
    }

    private void computeSubwords(String word, List<Integer> ngrams, List<String> substrings, BiConsumer<List<Integer>, Integer> pushMethod) {
        for (int i = 0; i < word.length(); i++) {
            if ((word.charAt(i) & 0xC0) == 0x80) continue;
            StringBuilder ngram = new StringBuilder();
            for (int j = i, n = 1; j < word.length() && n <= args.maxn(); n++) {
                ngram.append(word.charAt(j++));
                while (j < word.length() && (word.charAt(j) & 0xC0) == 0x80) {
                    ngram.append(word.charAt(j++));
                }
                if (n >= args.minn() && !(n == 1 && (i == 0 || j == word.length()))) {
                    int h = (int) (hash(ngram.toString()) % args.bucket());
                    pushMethod.accept(ngrams, h);
                    if (substrings != null) {
                        substrings.add(ngram.toString());
                    }
                }
            }
        }
    }

    /**
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
     * @param hashes
     * @param id
     */
    private void pushHash(List<Integer> hashes, int id) {
        if (pruneidx_size_ == 0 || id < 0) return;
        if (pruneidx_size_ > 0) {
            if (pruneidx_.containsKey(id)) {
                id = pruneidx_.get(id);
            } else {
                return;
            }
        }
        hashes.add(nwords_ + id);
    }

    /**
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
     * @param reader
     * @param logs
     * @param logLevel
     * @throws IOException
     * @throws IllegalStateException
     */
    void readFromFile(FTReader reader, PrintLogs logs, int logLevel) throws IOException, IllegalStateException {
        long minThreshold = 1;
        String word;
        while ((word = readWord(reader)) != null) {
            add(word);
            if (ntokens_ % 1_000_000 == 0 && logLevel > 1) {
                logs.printf("\rRead %dM words", ntokens_ / 1_000_000);
            }
            if (size_ > 0.75 * MAX_VOCAB_SIZE) {
                minThreshold++;
                threshold(minThreshold, minThreshold);
            }
        }
        threshold(this.args.minCount(), this.args.minCountLabel());
        initTableDiscard();
        initNgrams();
        if (logLevel > 0) {
            logs.printf("\rRead %dM words\n", ntokens_ / 1_000_000);
            logs.println("Number of words:  " + nwords_);
            logs.println("Number of labels: " + nlabels_);
        }
        if (size_ == 0) {
            throw new IllegalStateException("Empty vocabulary. Try a smaller -minCount value.");
        }
    }

    /**
     * Original code:
     * <pre>{@code bool Dictionary::readWord(std::istream& in, std::string& word) const {
     *  char c;
     *  std::streambuf& sb = *in.rdbuf();
     *  word.clear();
     *  while ((c = sb.sbumpc()) != EOF) {
     *      if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f' || c == '\0') {
     *          if (word.empty()) {
     *              if (c == '\n') {
     *                  word += EOS;
     *                  return true;
     *              }
     *              continue;
     *          } else {
     *              if (c == '\n')
     *                  sb.sungetc();
     *              return true;
     *          }
     *      }
     *      word.push_back(c);
     *  }
     *  in.get();
     *  return !word.empty();
     * }
     * }</pre>
     *
     * @param reader, {@link Reader} markSupported = true.
     * @return String or null if the end of stream
     * @throws IOException if something is wrong
     */
    public static String readWord(Reader reader) throws IOException {
        StringBuilder sb = new StringBuilder();
        int i;
        while ((i = reader.read()) != -1) {
            char c = (char) i;
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == 0x000b || c == '\f' || c == '\0') {
                if (sb.length() == 0) {
                    if (c == '\n') {
                        sb.append(EOS);
                        return sb.toString();
                    }
                    continue;
                } else {
                    if (c == '\n') {
                        reader.reset();
                    }
                    return sb.toString();
                }
            }
            reader.mark(1);
            sb.append(c);
        }
        return sb.length() == 0 ? null : sb.toString();
    }

    /**
     * <pre>{@code void Dictionary::initTableDiscard() {
     *  pdiscard_.resize(size_);
     *  for (size_t i = 0; i < size_; i++) {
     *      real f = real(words_[i].count) / real(ntokens_);
     *      pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
     *  }
     * }}</pre>
     */
    private void initTableDiscard() {
        pdiscard_ = new ArrayList<>(size_);
        for (int i = 0; i < size_; i++) {
            float f = ((float) words_.get(i).count) / ntokens_;
            pdiscard_.add((float) (Math.sqrt(args.samplingThreshold() / f) + args.samplingThreshold() / f));
        }
    }

    /**
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
     * @param type
     * @return
     */
    public List<Long> getCounts(EntryType type) {
        List<Long> counts = new ArrayList<>(EntryType.LABEL == type ? nlabels() : nwords());
        for (Entry w : words_) {
            if (w.type == type)
                counts.add(w.count);
        }
        return counts;
    }

    /**
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
     * @param in {@link FTReader}
     * @param words List of words
     * @param labels List of labels
     * @return int32_t
     * @throws IOException if an I/O error occurs
     */
    public int getLine(FTReader in, List<Integer> words, List<Integer> labels) throws IOException {
        List<Integer> word_hashes = new ArrayList<>();
        int ntokens = 0;
        reset(in);
        words.clear();
        labels.clear();
        String token;
        while ((token = readWord(in)) != null) {
            long h = hash(token);
            int wid = getId(token, h);
            EntryType type = wid < 0 ? getType(token) : getType(wid);
            ntokens++;
            if (EntryType.WORD == type) {
                addSubwords(words, token, wid);
                word_hashes.add((int) h);
            } else if (type == EntryType.LABEL && wid >= 0) {
                labels.add(wid - nwords_);
            }
            if (Objects.equals(token, EOS)) break;
        }
        addWordNgrams(words, word_hashes, args.wordNgrams());
        return ntokens;
    }

    /**
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
     * @param in {@link FTReader}
     * @param words List of words
     * @param rng {@link RandomGenerator}
     * @return int32_t
     * @throws IOException if an I/O error occurs
     */
    public int getLine(FTReader in, List<Integer> words, RandomGenerator rng) throws IOException {
        UniformRealDistribution uniform = new UniformRealDistribution(rng, 0, 1);
        int ntokens = 0;
        reset(in);
        words.clear();
        String token;
        while ((token = readWord(in)) != null) {
            long h = hash(token);
            int wid = getId(token, h);
            if (wid < 0) continue;
            ntokens++;
            if (EntryType.WORD == getType(wid) && !discard(wid, (float) uniform.sample())) {
                words.add(wid);
            }
            if (ntokens > MAX_LINE_SIZE || Objects.equals(token, EOS)) break;
        }
        return ntokens;
    }

    /**
     * <pre>{@code
     * bool Dictionary::discard(int32_t id, real rand) const {
     *  assert(id >= 0);
     *  assert(id < nwords_);
     *  if (args_->model == model_name::sup) return false;
     *      return rand > pdiscard_[id];
     * }
     * }</pre>
     *
     * @param id
     * @param rand
     * @return
     */
    private boolean discard(int id, float rand) {
        Validate.isTrue(id >= 0);
        Validate.isTrue(id < nwords_);
        return args.model() != Args.ModelName.SUP && rand > pdiscard_.get(id);
    }

    private static final long ADD_WORDS_NGRAMS_FACTOR_LONG = 116_049_371L;
    private static final UnsignedLong ADD_WORDS_NGRAMS_FACTOR_UNSIGNED_LONG = UnsignedLong.valueOf(ADD_WORDS_NGRAMS_FACTOR_LONG);

    /**
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
     * @param line
     * @param hashes
     * @param n
     */
    public void addWordNgrams(List<Integer> line, List<Integer> hashes, int n) {
        UnsignedLong bucket = UnsignedLong.valueOf(args.bucket());
        for (int i = 0; i < hashes.size(); i++) { // int32_t
            UnsignedLong h = UnsignedLong.fromLongBits(hashes.get(i)); // uint64_t
            for (int j = i + 1; j < hashes.size() && j < i + n; j++) { // h = h * 116049371 + hashes[j] :
                h = h.times(ADD_WORDS_NGRAMS_FACTOR_UNSIGNED_LONG).plus(UnsignedLong.fromLongBits(hashes.get(j)));
                pushHash(line, h.mod(bucket).intValue()); // h % args_->bucket
            }
        }
    }

    /**
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
     * @param line
     * @param token
     * @param wid
     */
    private void addSubwords(List<Integer> line, String token, int wid) {
        if (wid < 0) { // out of vocab
            computeSubwords(BOW + token + EOW, line);
        } else {
            if (args.maxn() <= 0) { // in vocab w/o subwords
                line.add(wid);
            } else { // in vocab w/ subwords
                List<Integer> ngrams = getSubwords(wid);
                line.addAll(ngrams);
            }
        }
    }

    /**
     * <pre>{@code
     * const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
     *  assert(i >= 0);
     *  assert(i < nwords_);
     *  return words_[i].subwords;
     * }
     * }</pre>
     *
     * @param i
     * @return
     */
    List<Integer> getSubwords(int i) {
        Validate.isTrue(i >= 0);
        Validate.isTrue(i < nwords_);
        return words_.get(i).subwords;
    }

    /**
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
    List<Integer> getSubwords(String word) {
        int i = getId(word);
        if (i >= 0) {
            return getSubwords(i);
        }
        List<Integer> ngrams = new ArrayList<>();
        computeSubwords(BOW + word + EOW, ngrams);
        return ngrams;
    }

    /**
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
    Multimap<String, Integer> getSubwordsMap(String word) {
        List<Integer> ngrams = new ArrayList<>();
        List<String> substrings = new ArrayList<>();
        int i = getId(word);
        if (i >= 0) {
            ngrams.add(i);
            substrings.add(words_.get(i).word);
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
     * <pre>{@code
     * void Dictionary::reset(std::istream& in) const {
     *  if (in.eof()) {
     *      in.clear();
     *      in.seekg(std::streampos(0));
     *  }
     * }
     * }</pre>
     *
     * @param in
     */
    public static void reset(FTReader in) throws IOException {
        if (!in.end()) {
            return;
        }
        in.rewind();
    }

    /**
     * <pre>{@code
     * std::string Dictionary::getLabel(int32_t lid) const {
     *  if (lid < 0 || lid >= nlabels_) {
     *      throw std::invalid_argument("Label id is out of range [0, " + std::to_string(nlabels_) + "]");
     *  }
     *  return words_[lid + nwords_].word;
     * }}</pre>
     *
     * @param lid
     * @return
     */
    public String getLabel(int lid) {
        if (lid < 0 || lid >= nlabels_) {
            throw new IllegalArgumentException("Label id is out of range [0, " + nlabels_ + "]");
        }
        return words_.get(lid + nwords_).word;
    }

    /**
     * <pre>{@code bool isPruned() {
     *  return pruneidx_size_ >= 0;
     *  }
     * }</pre>
     *
     * @return
     */
    public boolean isPruned() {
        return pruneidx_size_ >= 0;
    }

    /**
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
     * @param wordThreshold
     * @param labelThreshold
     */
    void threshold(long wordThreshold, long labelThreshold) {
        // todo: mb parallel stream ?
        ArrayList<Entry> words = words_.stream()
                .sorted(ENTRY_COMPARATOR) // todo: why?
                .filter(e -> (EntryType.WORD != e.type || e.count >= wordThreshold) && (EntryType.LABEL != e.type || e.count >= labelThreshold))
                .collect(Collectors.toCollection(ArrayList::new));
        words.trimToSize();
        this.words_ = words;
        this.word2int_ = new HashMap<>(words.size());
        int wordsCount = 0;
        int labelsCount = 0;
        int count = 0;
        for (Entry e : words) { // todo: move to one cycle?
            long h = find(e.word);
            word2int_.put(h, count++);
            if (EntryType.WORD == e.type) wordsCount++;
            if (EntryType.LABEL == e.type) labelsCount++;
        }
        this.size_ = words_.size();
        this.nwords_ = wordsCount;
        this.nlabels_ = labelsCount;
    }

    /**
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
     * @param idx
     * @return
     */
    List<Integer> prune(List<Integer> idx) {
        List<Integer> words = new ArrayList<>();
        List<Integer> ngrams = new ArrayList<>();
        for (Integer it : idx) {
            if (it < nwords_) {
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
                pruneidx_.put(ngram - nwords_, j);
                j++;
            }
            res.addAll(ngrams);
        }
        pruneidx_size_ = pruneidx_.size();
        word2int_.clear();
        int j = 0;
        for (int i = 0; i < words_.size(); i++) {
            if (getType(i) != EntryType.LABEL && (j >= words.size() || words.get(j) != i)) {
                continue;
            }
            words_.set(j, words_.get(i));
            word2int_.put(find(words_.get(j).word), j);
            j++;
        }
        nwords_ = words.size();
        size_ = nwords_ + nlabels_;
        words_ = words_.subList(0, size_);
        initNgrams();
        return res;
    }

    /**
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
        out.writeInt(size_);
        out.writeInt(nwords_);
        out.writeInt(nlabels_);
        out.writeLong(ntokens_);
        out.writeLong(pruneidx_size_);
        for (Entry e : words_) {
            FTOutputStream.writeString(out, e.word, charset);
            out.writeLong(e.count);
            out.writeByte(e.type.ordinal());
        }
        for (Integer key : pruneidx_.keySet()) {
            out.writeInt(key);
            out.writeInt(pruneidx_.get(key));
        }
    }

    /**
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
     * @param args
     * @param charset
     * @param in {@link FTInputStream}
     * @return {@link Dictionary} new instance
     * @throws IOException if an I/O error occurs
     */
    static Dictionary load(Args args, Charset charset, FTInputStream in) throws IOException {
        Dictionary res = new Dictionary(args, charset);
        res.size_ = in.readInt();
        res.nwords_ = in.readInt();
        res.nlabels_ = in.readInt();
        res.ntokens_ = in.readLong();
        res.pruneidx_size_ = in.readLong();
        res.word2int_ = new HashMap<>(res.size_);
        res.words_ = new ArrayList<>(res.size_);
        for (int i = 0; i < res.size_; i++) {
            Entry e = new Entry(FTInputStream.readString(in, res.charset), in.readLong(), EntryType.fromValue(in.readByte()));
            res.words_.add(e);
            res.word2int_.put(res.find(e.word), i);
        }
        res.pruneidx_.clear();
        for (int i = 0; i < res.pruneidx_size_; i++) {
            res.pruneidx_.put(in.readInt(), in.readInt());
        }
        res.initTableDiscard();
        res.initNgrams();
        return res;
    }

    Dictionary copy() {
        Dictionary res = new Dictionary(args, charset);
        res.size_ = this.size_;
        res.nwords_ = this.nwords_;
        res.nlabels_ = this.nlabels_;
        res.ntokens_ = this.ntokens_;
        res.pruneidx_size_ = this.pruneidx_size_;
        res.word2int_ = this.word2int_;
        res.word2int_ = new HashMap<>(this.word2int_);
        res.words_ = new ArrayList<>(this.words_.size());
        this.words_.forEach(entry -> res.words_.add(entry.copy()));
        res.words_ = new ArrayList<>(this.words_);
        res.pruneidx_ = new HashMap<>(this.pruneidx_);
        res.pdiscard_ = new ArrayList<>(this.pdiscard_);
        return res;
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
