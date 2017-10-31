package cc.fasttext;

import java.io.IOException;
import java.util.Random;

import ru.avicomp.io.FTInputStream;
import ru.avicomp.io.FTOutputStream;

/**
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.cc'>matrix.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.h'>matrix.h</a>
 */
public strictfp class Matrix {

    public float[][] data_;
    public int m_ = 0; // vocabSize
    public int n_ = 0; // layer1Size

    public Matrix() {
    }

    public Matrix(int m, int n) {
        m_ = m;
        n_ = n;
        data_ = new float[m][n];
    }

    public Matrix(final Matrix other) {
        m_ = other.m_;
        n_ = other.n_;
        data_ = new float[m_][n_];
        for (int i = 0; i < m_; i++) {
            System.arraycopy(other.data_[i], 0, data_[i], 0, n_);
        }
    }

    public void zero() {
        data_ = new float[m_][n_];
        /*for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                data_[i][j] = 0f;
            }
        }*/
    }

    /**
     * <pre>{@code
     * void Matrix::uniform(real a) {
     *  std::minstd_rand rng(1);
     *  std::uniform_real_distribution<> uniform(-a, a);
     *  for (int64_t i = 0; i < (m_ * n_); i++) {
     *      data_[i] = uniform(rng);
     *  }
     * }
     * }</pre>
     *
     * @param a
     */
    public void uniform(float a) {
        Random random = new Random(1L);
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                data_[i][j] = Utils.randomFloat(random, -a, a);
            }
        }
    }

    public float dotRow(final Vector vec, int i) {
        Utils.checkArgument(i >= 0);
        Utils.checkArgument(i < m_);
        Utils.checkArgument(vec.m_ == n_);
        float d = 0f;
        for (int j = 0; j < n_; j++) {
            d += data_[i][j] * vec.data_[j];
        }
        return d;
    }

    /**
     * <pre>{@code void Matrix::addRow(const Vector& vec, int64_t i, real a) {
     *  assert(i >= 0);
     *  assert(i < m_);
     *  assert(vec.size() == n_);
     *  for (int64_t j = 0; j < n_; j++) {
     *      data_[i * n_ + j] += a * vec.data_[j];
     *  }
     * }}</pre>
     *
     * @param vec
     * @param i
     * @param a
     */
    public void addRow(final Vector vec, int i, float a) {
        Utils.checkArgument(i >= 0);
        Utils.checkArgument(i < m_);
        Utils.checkArgument(vec.m_ == n_);
        for (int j = 0; j < n_; j++) {
            data_[i][j] += a * vec.data_[j];
        }
    }

    /**
     * <pre>{@code
     * void Matrix::save(std::ostream& out) {
     *  out.write((char*) &m_, sizeof(int64_t));
     *  out.write((char*) &n_, sizeof(int64_t));
     *  out.write((char*) data_, m_ * n_ * sizeof(real));
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    void save(FTOutputStream out) throws IOException {
        out.writeLong(m_);
        out.writeLong(n_);
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                out.writeFloat(data_[i][j]);
            }
        }
    }

    /**
     * <pre>{@code void Matrix::load(std::istream& in) {
     *  in.read((char*) &m_, sizeof(int64_t));
     *  in.read((char*) &n_, sizeof(int64_t));
     *  delete[] data_;
     *  data_ = new real[m_ * n_];
     *  in.read((char*) data_, m_ * n_ * sizeof(real));
     * }}</pre>
     *
     * @param in {@link FTInputStream}
     * @throws IOException if an I/O error occurs
     */
    void load(FTInputStream in) throws IOException {
        m_ = (int) in.readLong();
        n_ = (int) in.readLong();
        data_ = new float[m_][n_];
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < m_; j++) {
                data_[i][j] = in.readFloat();
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Matrix [data_=");
        if (data_ != null) {
            builder.append("[");
            for (int i = 0; i < m_ && i < 10; i++) {
                for (int j = 0; j < n_ && j < 10; j++) {
                    builder.append(data_[i][j]).append(",");
                }
            }
            builder.setLength(builder.length() - 1);
            builder.append("]");
        } else {
            builder.append("null");
        }
        builder.append(", m_=");
        builder.append(m_);
        builder.append(", n_=");
        builder.append(n_);
        builder.append("]");
        return builder.toString();
    }

}
