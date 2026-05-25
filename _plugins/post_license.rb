Jekyll::Hooks.register :posts, :post_render do |post|
  license = post.data["license"]
  next unless license.is_a?(Hash)

  name = license["name"]
  url = license["url"]
  notice = license["notice"]
  next if name.to_s.empty? || url.to_s.empty? || notice.to_s.empty?

  marker = 'data-post-license-notice="true"'
  next if post.output.include?(marker)

  license_html = <<~HTML
    <section class="post-license-notice" #{marker} style="margin-top: 2rem; padding: 1rem 1.25rem; border: 1px solid rgba(127, 127, 127, 0.25); border-radius: 0.75rem; background: rgba(127, 127, 127, 0.06);">
      <h2 style="margin-top: 0; font-size: 1rem;">License</h2>
      <p style="margin-bottom: 0.5rem;">
        <a href="#{url}" rel="license noopener noreferrer">#{name}</a>
      </p>
      <p style="margin-bottom: 0;">#{notice}</p>
    </section>
  HTML

  post.output = "#{post.output}#{license_html}"
end
