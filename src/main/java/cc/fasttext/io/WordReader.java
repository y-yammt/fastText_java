package cc.fasttext.io;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * The buffered stream reader which allows to read word-tokens from any binary {@link InputStream input stream}.
 * Not thread-safe.
 * Must be fast.
 * <p>
 * Created by @szuev on 20.12.2017.
 */
public class WordReader implements Closeable {
    public static final int END = -129;
    private final Charset charset;
    private final InputStream in;
    private final String newLine;
    private final byte[] delimiters;
    private final byte[] buffer;

    private int index;
    private int res;
    private byte[] tmp;
    private int start;

    /**
     * Main constructor.
     *
     * @param in            {@link InputStream} the input stream to wrap
     * @param charset       {@link Charset} encoding
     * @param bufferSize    the size of buffer
     * @param newLineSymbol String, to return on new line
     * @param delimiters    sequence of delimiters as byte array, not empty, the first element will be treated as new line symbol (e.g. '\n')
     */
    public WordReader(InputStream in, Charset charset, int bufferSize, String newLineSymbol, byte... delimiters) {
        this.charset = Objects.requireNonNull(charset, "Null charset");
        this.in = Objects.requireNonNull(in, "Null input stream");
        if (bufferSize <= 0) {
            throw new IllegalArgumentException("Buffer size must be positive number");
        }
        this.buffer = new byte[bufferSize];
        this.newLine = Objects.requireNonNull(newLineSymbol, "New line symbol can not be empty");
        if (delimiters.length == 0) {
            throw new IllegalArgumentException("No delimiters specified.");
        }
        this.delimiters = delimiters;
    }

    public WordReader(InputStream in, Charset charset, int bufferSize, String newLineSymbol, String delimiters) {
        this(in, charset, bufferSize, newLineSymbol, Objects.requireNonNull(delimiters, "Null delimiters").getBytes(charset));
    }

    public WordReader(InputStream in) {
        this(in, StandardCharsets.UTF_8, 8 * 1024, "\n", "\n ");
    }

    /**
     * Reads the next byte of data from the underling input stream.
     *
     * @return byte, int from -128 to 127 or {@link #END} in case of end of stream.
     * @throws IOException if some I/O error occurs
     * @see InputStream#read(byte[], int, int)
     */
    public int nextByte() throws IOException {
        if (index == buffer.length || res == 0) {
            if (start != 0) {
                if (buffer.length <= start) {
                    start = 0;
                } else {
                    tmp = new byte[buffer.length - start];
                    System.arraycopy(buffer, start, tmp, 0, tmp.length);
                }
            }
            res = read(buffer, 0, buffer.length);
            if (res == -1) {
                return END;
            }
            index = 0;
        }
        if (index < res) {
            return buffer[index++];
        }
        return END;
    }

    /**
     * Reads next word token from the underling input stream.
     * Original c++ code from dictionary.cc:
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
     * @return String or null in case of end of stream
     * @throws IOException if some I/O error occurs
     */
    public String nextWord() throws IOException {
        this.start = index;
        int len = 0;
        int b;
        while ((b = nextByte()) != END) {
            if (!isDelimiter(b)) {
                len++;
                continue;
            }
            if (len == 0) {
                if (isNewLine(b)) {
                    return newLine;
                }
                start++;
            } else {
                if (isNewLine(b)) {
                    --index;
                }
                return makeString(len);
            }
        }
        return len == 0 ? null : makeString(len);
    }

    protected boolean isDelimiter(int b) {
        for (byte i : delimiters) {
            if (b == i) return true;
        }
        return false;
    }

    protected boolean isNewLine(int b) {
        return delimiters[0] == b;
    }

    protected int read(byte[] array, int offset, int length) throws IOException {
        return in.read(array, offset, length);
    }

    private String makeString(int len) {
        String res;
        if (start > index) {
            byte[] bytes = new byte[len];
            if (len <= tmp.length) {
                System.arraycopy(tmp, 0, bytes, 0, len);
                this.start = start + len;
            } else {
                System.arraycopy(tmp, 0, bytes, 0, tmp.length);
                System.arraycopy(buffer, 0, bytes, tmp.length, len - tmp.length);
            }
            res = new String(bytes, charset);
        } else {
            res = new String(buffer, start, len, charset);
        }
        return res;
    }

    @Override
    public void close() throws IOException {
        in.close();
    }
}
